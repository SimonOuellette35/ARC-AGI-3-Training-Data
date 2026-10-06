"""Train Olmo on policy tensors, tokenizing only distinct output programs."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from utils.goal_hypothesizer import add_training_arguments
from utils.goal_prediction import add_prediction_arguments


def train(args):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainingArguments, set_seed)
    from utils.goal_numeric_data import NumericGoalDataset, collate_numeric
    from utils.goal_numeric_model import NumericGoalModel
    from utils.goal_prediction import make_predictor
    from utils.policy_checkpoint import policy_from_checkpoint
    from utils.goal_numeric_checkpoint import save_state, load_state
    from utils.training_resume import keep_training_config

    if args.preprocess_only:
        raise ValueError('Numeric A has no input-tokenization stage. Run prepare, then train.')
    set_seed(args.seed)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.load_in_4bit and device != 'cuda':
        raise ValueError('--load-in-4bit requires CUDA')
    if device == 'cpu' and not args.allow_cpu:
        raise ValueError('CUDA unavailable; use --allow-cpu only for small diagnostics')
    checkpoint = torch.load(args.policy_ckpt, map_location='cpu', weights_only=False)
    policy, cfg = policy_from_checkpoint(checkpoint, device)
    del checkpoint
    states = args.max_states or cfg.max_states
    frames = args.max_frames or cfg.max_frames
    if not 0 < states <= cfg.max_states or not 0 < frames <= cfg.max_frames:
        raise ValueError('Input caps must be positive and fit the policy checkpoint')
    # Keep embedding sizes/checkpoint config intact; use smaller caps only for packing.
    packing_cfg = replace(cfg, max_states=states, max_frames=frames)
    dataset = NumericGoalDataset(args.data_dir, 'train', packing_cfg, cache_dir=args.cache_dir)
    if not len(dataset):
        raise ValueError('No training examples in numeric index')
    settings = dataset.settings
    prediction_args = argparse.Namespace(**settings)
    prediction_args.prediction_device = device
    predictor = make_predictor(prediction_args)
    dataset.predictor = predictor
    validation = NumericGoalDataset(args.data_dir, 'validation', packing_cfg, cache_dir=args.cache_dir)
    eval_dataset = validation if len(validation) else None
    metric = 'eval_loss'
    if predictor is not None and len(validation):
        eval_dataset = {'clean': validation, 'mixed': NumericGoalDataset(
            args.data_dir, 'validation', packing_cfg, predictor=predictor, cache_dir=args.cache_dir)}
        metric = 'eval_mixed_loss'

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.eos_token_id is None:
        raise ValueError('Decoder tokenizer must define EOS')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if device == 'cpu' else (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    kwargs = dict(revision=args.revision, torch_dtype=dtype, attn_implementation=args.attn_implementation)
    if args.load_in_4bit:
        kwargs.update(device_map={'': torch.cuda.current_device()}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype))
    decoder = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.max_length > decoder.config.max_position_embeddings:
        raise ValueError('--max-length exceeds the Olmo context window')
    decoder.config.use_cache = False
    if args.load_in_4bit:
        decoder = prepare_model_for_kbit_training(decoder, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False})
    else:
        decoder.to(device)
    decoder = get_peft_model(decoder, LoraConfig(task_type='CAUSAL_LM', r=args.lora_r,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias='none', target_modules='all-linear'))
    model = NumericGoalModel(policy, decoder, tokenizer, args.max_length)
    model.projector.to(device)
    # Save the actual packing caps for runtime without resizing policy embeddings.
    model.packing_caps = dict(max_states=states, max_frames=frames)
    maximum_target = max(len(model.target_tokens(p)) for p in dataset.programs.values())
    per_frame = getattr(cfg, 'temporal_tokens', cfg.patch_grid ** 2)
    extras = 3 if getattr(cfg, 'architecture', 'packed_v1') == 'packed_v1' and cfg.dynamics else 2
    worst_case = states * (frames * per_frame + extras) - (extras - 1) + len(model.header) + maximum_target
    if worst_case > args.max_length:
        raise ValueError(f'Configured history plus largest program may require {worst_case} tokens; '
                         'increase --max-length or reduce --max-states/--max-frames before training')
    print(f'[train] {len(dataset):,} numeric examples; {len(dataset.programs):,} distinct programs; '
          f'no observation text/token cache; frozen policy encoder + trainable projection + Olmo LoRA', flush=True)

    class NumericTrainer(Trainer):
        def _save_optimizer_and_scheduler(self, output_dir):
            if self.args.should_save:
                save_state({'optimizer': self.optimizer.state_dict(), 'scheduler': self.lr_scheduler.state_dict()},
                           Path(output_dir) / 'numeric_optimizer')

        def _load_optimizer_and_scheduler(self, checkpoint):
            if checkpoint:
                state = load_state(Path(checkpoint) / 'numeric_optimizer')
                with keep_training_config(self.optimizer, self.lr_scheduler):
                    self.optimizer.load_state_dict(state['optimizer'])
                    self.lr_scheduler.load_state_dict(state['scheduler'])

        def _save_scaler(self, output_dir):
            if self.accelerator.scaler is not None and self.args.should_save:
                save_state(self.accelerator.scaler.state_dict(), Path(output_dir) / 'numeric_scaler')

        def _load_scaler(self, checkpoint):
            if checkpoint and self.accelerator.scaler is not None:
                self.accelerator.scaler.load_state_dict(load_state(Path(checkpoint) / 'numeric_scaler'))

        def _save_rng_state(self, output_dir):
            import numpy as np
            import random
            numpy_state = list(np.random.get_state())
            numpy_state[1] = torch.from_numpy(numpy_state[1].astype(np.int64))
            state = dict(python=random.getstate(), numpy=numpy_state, cpu=torch.get_rng_state())
            if torch.cuda.is_available():
                state['cuda'] = torch.cuda.get_rng_state_all()
            save_state(state, Path(output_dir) / f'numeric_rng_{self.args.process_index}')

        def _load_rng_state(self, checkpoint):
            import numpy as np
            import random
            if checkpoint:
                state = load_state(Path(checkpoint) / f'numeric_rng_{self.args.process_index}')
                random.setstate(state['python'])
                state['numpy'][1] = state['numpy'][1].numpy().astype(np.uint32)
                np.random.set_state(tuple(state['numpy']))
                torch.set_rng_state(state['cpu'])
                if torch.cuda.is_available() and 'cuda' in state:
                    torch.cuda.set_rng_state_all(state['cuda'])

        def _save(self, output_dir=None, state_dict=None):
            output_dir = Path(output_dir or self.args.output_dir)
            self.model.save_pretrained(output_dir)
            torch.save(self.args, output_dir / 'training_args.bin')

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            from peft import set_peft_model_state_dict
            from peft.utils.save_and_load import load_peft_weights
            model = model or self.model
            root = Path(resume_from_checkpoint)
            metadata = json.loads((root / 'numeric_config.json').read_text())
            if (metadata['max_length'] != model.max_length or
                    metadata.get('packing_caps') != model.packing_caps):
                raise ValueError('Resume requires unchanged numeric input/context settings')
            saved = torch.load(root / 'numeric_encoder.pt', map_location='cpu', weights_only=False)
            if saved['cfg'] != vars(model.policy.cfg):
                raise ValueError('Resume policy architecture/config changed')
            model.policy.load_state_dict(saved['model'])
            model.projector.load_state_dict(saved['projector'])
            set_peft_model_state_dict(model.decoder, load_peft_weights(str(root), device='cpu'))

        def _load_best_model(self):
            self._load_from_checkpoint(self.state.best_model_checkpoint)

    training_args = TrainingArguments(output_dir=str(args.output_dir),
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, lr_scheduler_type='cosine', warmup_ratio=.03,
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16, use_cpu=device == 'cpu',
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=10, save_steps=args.save_steps, eval_steps=args.save_steps,
        eval_strategy='steps' if eval_dataset is not None else 'no',
        save_total_limit=2, load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model=metric, greater_is_better=False, prediction_loss_only=True,
        report_to='none', seed=args.seed, remove_unused_columns=False, label_names=[],
        dataloader_num_workers=0, dataloader_pin_memory=device == 'cuda')
    trainer = NumericTrainer(model=model, args=training_args, train_dataset=dataset,
        eval_dataset=eval_dataset, data_collator=collate_numeric)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir / 'adapter'))
    (args.output_dir / 'run_config.json').write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2) + '\n')


def main():
    from utils.goal_numeric_data import prepare
    parser = argparse.ArgumentParser(description='Hypothesizer A: policy numeric inputs -> Olmo LoRA is_win program')
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Index existing policy episodes and program labels; no copied observations')
    prep.add_argument('--cache-dir', type=Path, default=Path('data/cache_policy'))
    prep.add_argument('--goal-root', type=Path, default=Path('data/training_goal_data'))
    prep.add_argument('--demos-root', type=Path, default=Path('data/training_multi_level'))
    prep.add_argument('--output-dir', type=Path, default=Path('data/goal_numeric_A'))
    prep.add_argument('--resume', action='store_true')
    prep.add_argument('--games', nargs='+')
    prep.add_argument('--max-episodes-per-game', type=int, default=0)
    prep.add_argument('--samples-per-level', type=int, default=3)
    prep.add_argument('--val-fraction', type=float, default=.1)
    prep.add_argument('--no-action-shuffle', action='store_true')
    prep.add_argument('--seed', type=int, default=42)
    add_prediction_arguments(prep)
    fit = commands.add_parser('train', help='Train Olmo LoRA and the numeric-input projection')
    add_training_arguments(fit, data_dir='data/goal_numeric_A', output_dir='runs/goal_numeric_A',
                           input_tokenization=False)
    fit.add_argument('--preprocess-only', action='store_true', help=argparse.SUPPRESS)
    fit.add_argument('--policy-ckpt', type=Path, default=Path('runs/policy_dynamics_V1/best.pt'))
    fit.add_argument('--cache-dir', type=Path, help='Override indexed policy cache root')
    fit.add_argument('--max-states', type=int, help='Default: policy checkpoint cap')
    fit.add_argument('--max-frames', type=int, help='Default: policy checkpoint cap')
    fit.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    fit.add_argument('--allow-cpu', action='store_true', help='Permit a small CPU diagnostic')
    fit.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else train)(args)
