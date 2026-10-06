"""Shared label parsing, prompts, and LoRA training for goal hypothesizers.

Entry points supply their record tokenizer and cache format explicitly.
Training dependencies are imported lazily so data preparation works on CPU.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re

from utils.training_resume import CurrentConfigTrainerMixin
from utils.goal_prepared_data import prepared_path

MODEL_ID = "allenai/Olmo-3-7B-Think"
SYSTEM = "Infer the game's win rule from session evidence. Return only Python source defining is_win, without Markdown, reasoning, or <think> tags."
TASK = """Write def is_win(frame0, current_state, action, next_state) -> bool.
Return True exactly when the transition wins. All grids are 64x64 integers 0-15.
frame0 is the current level's initial settled frame, even after later RESETs.
current_state and next_state are one action apart; action is None or
{'id': int, 'data': dict}, with click data {'x': column, 'y': row}.
numpy is available as np and utils.goal_verifier as gv; do not import them.
Use gv helpers where appropriate. Output the complete program alone."""
# Keep Olmo's chat delimiters, but do not prefill a reasoning block. Save this
# template with the adapter tokenizer so inference uses the same convention.
DIRECT_CHAT_TEMPLATE = """{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] }}{% if message['role'] == 'assistant' %}{{ eos_token }}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""


def read_programs(path):
    """Match test_goals.py level mapping, without executing label code."""
    lines = Path(path).read_text().splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith("def is_win(")]
    if not starts:
        raise ValueError(f"{path}: no is_win definitions")
    programs, any_header = {}, False
    for k, start in enumerate(starts):
        previous = starts[k - 1] if k else 0
        headers = re.findall(r"^#.*\(index\s+(\d+)\)", "".join(lines[previous:start]), re.M)
        index = int(headers[-1]) if headers else k
        any_header |= bool(headers)
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        block = lines[start:end]
        # The next level's descriptive header is not part of this target.
        while block and (not block[-1].strip() or block[-1].startswith('#')):
            block.pop()
        source = "".join(block).strip() + "\n"
        tree = ast.parse(source)
        fn = tree.body[0]
        if not isinstance(fn, ast.FunctionDef) or [a.arg for a in fn.args.args] != [
            "frame0", "current_state", "action", "next_state"
        ]:
            raise ValueError(f"{path}: incompatible is_win signature at index {index}")
        if index in programs:
            raise ValueError(f"{path}: duplicate level index {index}")
        programs[index] = source
    return programs, len(programs) == 1 and not any_header


def split_for_game(game, seed, val_fraction):
    value = int(hashlib.sha256(f'{seed}:{game}'.encode()).hexdigest()[:16], 16) / 2**64
    return 'validation' if value < val_fraction else 'train'


def collate_examples(examples, pad_token_id):
    import torch
    examples = [dict(input_ids=e['input_ids'], attention_mask=[1] * len(e['input_ids']),
                     labels=[-100] * e['prompt_length'] + e['input_ids'][e['prompt_length']:])
                if 'prompt_length' in e else e for e in examples]
    size = max(len(example['input_ids']) for example in examples)
    return {key: torch.tensor([example[key] + [padding] * (size - len(example[key]))
                              for example in examples], dtype=torch.long)
            for key, padding in [('input_ids', pad_token_id), ('attention_mask', 0), ('labels', -100)]}


def train(args, *, encode_record, data_format):
    """Train with the caller's record tokenizer and cache format identifier."""
    from functools import partial
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments, set_seed
    from utils.goal_token_cache import tokenized_dataset

    class CurrentConfigTrainer(CurrentConfigTrainerMixin, Trainer):
        pass

    class CurrentConfigCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            # TrainerState also saves intervals and batch size. Its step/epoch
            # progress is retained, but these settings belong to the current run.
            state.compute_steps(args, state.max_steps)
            state.train_batch_size = args.train_batch_size

    set_seed(args.seed)
    if not args.preprocess_only and not torch.cuda.is_available():
        raise ValueError('Training requires CUDA; use --preprocess-only for CPU tokenization')
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.chat_template = DIRECT_CHAT_TEMPLATE
    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer must define EOS')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    if args.max_length > config.max_position_embeddings:
        raise ValueError('--max-length exceeds model context window')
    datasets = {}
    for split in ('train', 'validation', 'validation_mixed'):
        path = prepared_path(args.data_dir, split)
        if path is None:
            if split == 'validation_mixed':
                continue
            raise ValueError(f'Missing prepared data: {args.data_dir / split}')
        dataset = tokenized_dataset(path, tokenizer, encode_record, args.max_length,
            cache_dir=args.data_dir / 'tokenized_chunks',
            identity=[args.model, args.revision, SYSTEM, TASK, data_format],
            workers=args.tokenize_workers, chunk_size=args.tokenize_chunk_size)
        if dataset is not None:
            datasets[split] = dataset
    if 'train' not in datasets or not len(datasets['train']):
        raise ValueError('No usable training examples')
    validation, best_metric = evaluation_datasets(datasets)
    if args.preprocess_only:
        print('[tokenize] complete; rerun without --preprocess-only to train using this cache', flush=True)
        return
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs = dict(config=config, revision=args.revision, torch_dtype=dtype,
                  attn_implementation=args.attn_implementation, device_map={'': local_rank})
    if args.load_in_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False})
    model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=args.lora_r,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias='none', target_modules='all-linear'))
    model.print_trainable_parameters()
    training_args = TrainingArguments(output_dir=str(args.output_dir),
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, lr_scheduler_type='cosine', warmup_ratio=0.03,
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=10, save_steps=args.save_steps, eval_steps=args.save_steps,
        eval_strategy='steps' if validation is not None else 'no',
        save_total_limit=2, load_best_model_at_end=validation is not None,
        metric_for_best_model=best_metric, greater_is_better=False,
        prediction_loss_only=True, report_to='none', seed=args.seed,
        ddp_find_unused_parameters=False, remove_unused_columns=False)
    trainer = CurrentConfigTrainer(model=model, args=training_args, train_dataset=datasets['train'],
        eval_dataset=validation, processing_class=tokenizer,
        callbacks=[CurrentConfigCallback()],
        data_collator=partial(collate_examples, pad_token_id=tokenizer.pad_token_id))
    if args.resume_from_checkpoint:
        print(f'[train] resuming from {args.resume_from_checkpoint} '
              '(config from current defaults/CLI)')
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir / 'adapter'))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir / 'adapter')
        (args.output_dir / 'run_config.json').write_text(json.dumps({
            k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, indent=2) + '\n')


def evaluation_datasets(datasets):
    """Expose clean/mixed losses and select on the training mixture when present."""
    validation = datasets.get('validation')
    mixed = datasets.get('validation_mixed')
    if mixed is not None and len(mixed):
        if validation is None or not len(validation):
            raise ValueError('Mixed validation requires usable clean validation examples')
        return {'clean': validation, 'mixed': mixed}, 'eval_mixed_loss'
    return validation, 'eval_loss'


def add_training_arguments(parser, *, data_dir='data/goal_lora', output_dir='runs/goal_lora',
                           input_tokenization=True):
    """Shared optimizer/model options for the A and B input variants."""
    parser.add_argument('--data-dir', type=Path, default=Path(data_dir))
    parser.add_argument('--output-dir', type=Path, default=Path(output_dir))
    parser.add_argument('--model', default=MODEL_ID)
    parser.add_argument('--revision', default='main')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--max-length', type=int, default=16384)
    if input_tokenization:
        parser.add_argument('--tokenize-workers', type=int, default=0,
                            help='CPU preprocessing workers; 0 uses up to 8 available CPUs')
        parser.add_argument('--tokenize-chunk-size', type=int, default=2000,
                            help='source rows per resumable token cache chunk (default: 2000)')
        parser.add_argument('--preprocess-only', action='store_true',
                            help='build/resume token caches on CPU, then exit without loading model weights')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=16)
    parser.add_argument('--epochs', type=float, default=3)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--save-steps', type=int, default=250)
    parser.add_argument('--attn-implementation', choices=['sdpa', 'flash_attention_2', 'eager'], default='sdpa')
    parser.add_argument('--resume-from-checkpoint',
                        help='resume weights/progress using current defaults/CLI; '
                             'model and LoRA dimensions must fit the saved weights')
