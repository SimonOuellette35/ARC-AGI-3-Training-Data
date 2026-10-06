"""Numeric policy history -> trainable projection -> Olmo LoRA code decoder."""
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch import nn

from utils.goal_hypothesizer import DIRECT_CHAT_TEMPLATE, SYSTEM, TASK
from utils.goal_numeric_data import FORMAT
from utils.policy_checkpoint import policy_from_checkpoint


class NumericGoalModel(nn.Module):
    def __init__(self, policy, decoder, tokenizer, max_length=16384):
        super().__init__()
        self.policy = policy.eval().requires_grad_(False)
        self.decoder, self.tokenizer = decoder, tokenizer
        self.config = decoder.config
        self.max_length = max_length
        width = decoder.get_input_embeddings().weight.shape[1]
        self.projector = nn.Sequential(nn.LayerNorm(policy.cfg.d_model),
                                       nn.Linear(policy.cfg.d_model, width), nn.GELU(), nn.Linear(width, width))
        tokenizer.chat_template = DIRECT_CHAT_TEMPLATE
        self.header = tokenizer.apply_chat_template([
            {'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': TASK}],
            tokenize=True, add_generation_prompt=True)
        self.targets = {}

    def train(self, mode=True):
        super().train(mode)
        self.policy.eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    @property
    def is_gradient_checkpointing(self):
        return self.decoder.is_gradient_checkpointing

    def target_tokens(self, source):
        # One entry per distinct program, never per episode/history.
        if source not in self.targets:
            self.targets[source] = self.tokenizer.encode(source, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        return self.targets[source]

    def _prefixes(self, inputs):
        with torch.no_grad():
            history, valid = self.policy(**inputs, return_history=True)
        memory = self.projector(history.to(self.projector[0].weight.dtype))
        embed = self.decoder.get_input_embeddings()
        header = embed(torch.tensor(self.header, device=memory.device))
        return [torch.cat((row[mask].to(header.dtype), header), dim=0) for row, mask in zip(memory, valid)]

    def forward(self, frames, act, xy, mask, nframes, frame_step, frame_slot, frame_mask,
                programs=None, return_loss=True):
        from torch.nn.utils.rnn import pad_sequence
        inputs = dict(frames=frames, act=act, xy=xy, mask=mask, nframes=nframes,
                      frame_step=frame_step, frame_slot=frame_slot, frame_mask=frame_mask)
        prefixes = self._prefixes(inputs)
        if programs is None or len(programs) != len(prefixes):
            raise ValueError('Training requires one complete target program per history')
        embed = self.decoder.get_input_embeddings()
        sequences, labels, masks = [], [], []
        for prefix, source in zip(prefixes, programs):
            ids = torch.tensor(self.target_tokens(source), device=prefix.device)
            if len(prefix) + len(ids) > self.max_length:
                raise ValueError('Numeric history plus complete program exceeds --max-length; '
                                 'increase it within the Olmo context limit or lower --max-states/--max-frames')
            sequences.append(torch.cat((prefix, embed(ids)), dim=0))
            labels.append(torch.cat((ids.new_full((len(prefix),), -100), ids)))
            masks.append(ids.new_ones(len(prefix) + len(ids)))
        return self.decoder(inputs_embeds=pad_sequence(sequences, batch_first=True),
                            attention_mask=pad_sequence(masks, batch_first=True),
                            labels=pad_sequence(labels, batch_first=True, padding_value=-100), use_cache=False)

    @torch.inference_mode()
    def generate_program(self, inputs, max_new_tokens=2048):
        prefixes = self._prefixes(inputs)
        if len(prefixes) != 1:
            raise ValueError('Program generation expects one history')
        prefix = prefixes[0][None]
        if prefix.shape[1] + max_new_tokens > self.max_length:
            raise ValueError('Numeric history and generation budget exceed the model context')
        output = self.decoder.generate(inputs_embeds=prefix,
            attention_mask=torch.ones(prefix.shape[:2], device=prefix.device, dtype=torch.long),
            do_sample=False, max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True)
        # With inputs_embeds and no input_ids, HF returns only generated IDs.
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.decoder.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        cfg = asdict(self.policy.cfg)
        torch.save({'cfg': cfg, 'architecture': cfg.get('architecture', 'packed_v1'),
                    'model': self.policy.state_dict(), 'projector': self.projector.state_dict()},
                   directory / 'numeric_encoder.pt')
        (directory / 'numeric_config.json').write_text(json.dumps(
            {'format': FORMAT, 'max_length': self.max_length,
             'packing_caps': getattr(self, 'packing_caps', dict(max_states=self.policy.cfg.max_states,
                                                              max_frames=self.policy.cfg.max_frames))}, indent=2) + '\n')


def load_numeric_adapter(directory, *, base_model=None, device='cpu', load_in_4bit=False):
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    directory = Path(directory)
    config_file = directory / 'numeric_config.json'
    if not config_file.exists():
        raise ValueError('Hypothesizer A requires a numeric-input Olmo adapter; this is a legacy text adapter')
    metadata = json.loads(config_file.read_text())
    if metadata['format'] != FORMAT:
        raise ValueError('Unknown numeric goal model format')
    adapter = PeftConfig.from_pretrained(directory, local_files_only=True)
    device = torch.device(device)
    dtype = torch.float32 if device.type == 'cpu' else (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    kwargs = dict(local_files_only=True, torch_dtype=dtype, device_map={'': str(device)})
    if load_in_4bit:
        if device.type != 'cuda':
            raise ValueError('4-bit inference requires CUDA')
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    decoder = AutoModelForCausalLM.from_pretrained(base_model or adapter.base_model_name_or_path, **kwargs)
    decoder = PeftModel.from_pretrained(decoder, directory, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    saved = torch.load(directory / 'numeric_encoder.pt', map_location='cpu', weights_only=False)
    policy, _ = policy_from_checkpoint(saved, device)
    model = NumericGoalModel(policy, decoder, tokenizer, metadata['max_length'])
    model.projector.load_state_dict(saved['projector'])
    model.projector.to(device)
    model.packing_caps = metadata['packing_caps']
    return model.eval()
