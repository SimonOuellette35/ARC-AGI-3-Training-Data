"""Bounded spatial caching and V2 inference for one observed history.

These helpers require fixed eval-mode weights. Clear/recreate caches after any
weight, device or precision change. Training and padded batches use model.forward.
"""
from collections import OrderedDict

import numpy as np
import torch

from policy_runtime import policy_inputs


class FrameCache:
    """Bounded per-model V2 frame encoding cache, keyed by actual pixels.

    Temporal embeddings are applied by the model after cached spatial features.
    Must be discarded when weights/device/precision change. Eval inference only.
    """
    def __init__(self, model, capacity=512):
        if capacity < 1:
            raise ValueError('Frame cache capacity must be positive')
        self.model = model
        self.capacity = capacity
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def clear(self):
        self.entries.clear()

    def attach(self, inp):
        if getattr(self.model.cfg, 'architecture', None) != 'spatial_temporal_v2':
            device = next(self.model.parameters()).device
            return {k: v.to(device) for k, v in inp.items()}
        if self.model.training or torch.is_grad_enabled():
            raise ValueError('Frame cache requires eval inference')
        if inp['frames'].device.type != 'cpu' or inp['frames'].shape[0] != 1:
            raise ValueError('Frame cache expects one history on CPU')
        # Build keys on CPU before transferring input to CUDA.
        flat = inp['frames'][0]
        keys = [frame.numpy().astype(np.uint8, copy=False).tobytes() for frame in flat]
        missing = list(dict.fromkeys(k for k in keys if k not in self.entries))
        self.hits += len(keys) - len(missing)
        self.misses += len(missing)
        device = next(self.model.parameters()).device
        if missing:
            boards = np.stack([np.frombuffer(k, dtype=np.uint8).reshape(64, 64) for k in missing])
            frames = torch.as_tensor(boards, dtype=torch.long, device=device)
            spatial, summaries = self.model.encode_frames(frames)
            for i, k in enumerate(missing):
                self.entries[k] = (spatial[i].clone(), summaries[i].clone())
        features = [self.entries[k] for k in keys]
        for k in keys:
            self.entries.move_to_end(k)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        moved = {k: v.to(device) for k, v in inp.items()}
        moved['frame_features'] = tuple(torch.stack([v[i] for v in features])[None] for i in (0, 1))
        return moved


class SingleHistoryRuntime:
    def __init__(self, model, cache=None):
        if model.cfg.architecture != 'spatial_temporal_v2':
            raise ValueError('SingleHistoryRuntime requires V2')
        self.model=model
        self.device=next(model.parameters()).device
        self.cache=cache if cache is not None else FrameCache(model)
        self.layouts=OrderedDict()

    def layout(self, nframes):
        key=tuple(map(int,nframes))
        if key not in self.layouts:
            nf=np.asarray(nframes)
            k=self.model.cfg.temporal_tokens
            sizes=nf*k+2
            starts=np.cumsum(sizes)-sizes
            readout=starts+nf*k
            frame_at=np.repeat(starts,nf)+np.concatenate([np.arange(n)*k for n in nf])
            def tensor(x):return torch.as_tensor(x,dtype=torch.long,device=self.device)
            length=int(sizes.sum())
            self.layouts[key]=dict(length=length,readout=tensor(readout),action=tensor(readout+1),
                frame=tensor((frame_at[:,None]+np.arange(k)).reshape(-1)),
                steps=tensor(np.arange(len(nf))),last_action=tensor([length-1]),
                causal=torch.ones(length,length,dtype=torch.bool,device=self.device).triu(1))
        result=self.layouts[key]
        self.layouts.move_to_end(key)
        while len(self.layouts)>64:self.layouts.popitem(last=False)
        return result

    @torch.inference_mode()
    def forward(self, spans, actions, coords, move=None, predict=True):
        m=self.model
        if m.training:raise ValueError('Runtime requires eval mode')
        inp_cpu=policy_inputs(m.cfg,spans,actions,coords,torch.device('cpu'))
        layout=self.layout(inp_cpu['nframes'][0].numpy())
        if move is not None:
            inp_cpu['act'][0,-1]=move['idx']
            inp_cpu['xy'][0,-1]=torch.tensor(move['xy'] if move['xy'] is not None else (255,255))
        with torch.autocast(self.device.type,enabled=self.device.type=='cuda'):
            inp=self.cache.attach(inp_cpu)
            spatial,tokens=inp['frame_features']
            tokens=(tokens+m.step_pos(inp['frame_step']).unsqueeze(2)
                    +m.frame_pos(inp['frame_slot']).unsqueeze(2)+m.type_embed.weight[0])
            step=m.step_pos(layout['steps'])[None]
            readout=m.readout+m.type_embed.weight[1]+step
            action=m.action_io.action_token(inp['act'],inp['xy'])+m.type_embed.weight[2]+step
            seq=tokens.new_zeros(1,layout['length'],m.cfg.d_model)
            seq[0,layout['frame']]=tokens.reshape(-1,m.cfg.d_model)
            seq[0,layout['readout']]=readout[0].to(seq.dtype)
            seq[0,layout['action']]=action[0].to(seq.dtype)
            h=m.in_norm(seq)
            for layer in m.temporal_layers:
                h=layer(h,src_mask=layout['causal'],is_causal=True)
            h=m.out_norm(h)
            types=pointers=None
            if predict:
                policy=h[:,layout['readout'][-1]]
                types,pointers=m.action_io.predict(policy)
                if m.cfg.spatial_pointer:
                    pointers=pointers+m.spatial_pointer(policy,spatial[:,-1]).to(pointers.dtype)
            decoded=None
            if move is not None:
                decoded=m.dynamics_decoder(spatial[:,-1],h,layout['last_action'])
        return types,pointers,decoded,inp

