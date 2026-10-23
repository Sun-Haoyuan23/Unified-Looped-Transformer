"""One canonical Show-o2 runtime for capture and every same-state continuation."""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def rng_snapshot():
    return {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(),
            'python': random.getstate()}


def rng_restore(snapshot):
    torch.set_rng_state(snapshot['cpu'])
    torch.cuda.set_rng_state(snapshot['cuda'])
    random.setstate(snapshot['python'])


class Runtime:
    def __init__(self, work, steps=50, cfg=5.0, vae=True):
        self.work = Path(work)
        self.source = self.work / 'Show-o/show-o2'
        sys.path.insert(0, str(self.source))
        sys.path.insert(0, str(self.work/'stage2b'))
        from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
        from models.misc import get_text_tokenizer, prepare_gen_input
        from transport import Sampler, create_transport
        from utils import get_hyper_params
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Expose one healthy GPU per worker by UUID')
        self.device, self.dtype = torch.device('cuda:0'), torch.bfloat16
        self.cfg, self.steps = float(cfg), steps
        self.config = OmegaConf.load(self.source/'configs/showo2_1.5b_demo_432x432.yaml')
        self.tokenizer, self.ids = get_text_tokenizer(
            str(self.work/'models/qwen2.5-1.5b-tokenizer'), add_showo_tokens=True,
            return_showo_token_ids=True, llm_name='qwen2_5')
        self.model = Showo2Qwen2_5.from_pretrained(
            str(self.work/'models/show-o2-1.5B-local'), use_safetensors=False
        ).to(device=self.device, dtype=self.dtype).eval().requires_grad_(False)
        if any(p.is_meta for p in self.model.parameters()):
            raise RuntimeError('Unmaterialized model parameters')
        # Released t2i_generate is decorated with torch.no_grad. The unchanged
        # underlying body is needed for terminal training through frozen dynamics.
        body=getattr(self.model.t2i_generate,'__wrapped__',None)
        if body is None:raise RuntimeError('Released generation body cannot be unwrapped for rollout gradients')
        self.generation_body=body.__get__(self.model,type(self.model))
        self.original = list(self.model.diffusion_head_a)
        if len(self.original) != 10:
            raise RuntimeError('This stage requires the released original 10-block head')
        if self.config.model.showo.add_time_embeds:
            for name in ('num_t2i_image_tokens','num_mmu_image_tokens','num_video_tokens'):
                self.config.dataset.preprocessing[name] += 1
        h = get_hyper_params(self.config, self.tokenizer, self.ids)
        (self.nvisual,self.nmmu,_nv,self.maxlen,self.maxtext,self.channels,self.patch,
         self.w,self.h,self.pad,self.bos,self.eos,self.boi,self.eoi,
         _bov,_eov,self.imgpad,_vid,_guidance) = h
        self.mask_fn, self.prepare_fn = omni_attn_mask_naive, prepare_gen_input
        tr = self.config.transport
        transport = create_transport(path_type=tr.path_type,prediction=tr.prediction,
            loss_weight=tr.loss_weight,train_eps=tr.train_eps,sample_eps=tr.sample_eps,
            snr_type=tr.snr_type,do_shift=tr.do_shift,seq_len=self.nvisual)
        self.sample_fn = Sampler(transport).sample_ode(sampling_method=tr.sampling_method,
            num_steps=steps,atol=tr.atol,rtol=tr.rtol,reverse=tr.reverse,
            time_shifting_factor=tr.time_shifting_factor)
        integrator = self.sample_fn.__self__
        if integrator.sampler_type != 'euler' or integrator.do_shift:
            raise RuntimeError('Stage-3 replay currently supports this exact fixed Euler grid only')
        self.grid = integrator.t.to(self.device)
        self.drift = integrator.drift
        self.vae = WanVAE(vae_pth='/home/shy_23/models/show-o2-deps/Wan2.1_VAE.pth',
                          dtype=self.dtype,device=self.device) if vae else None

    def inputs(self, prompt):
        cond,uncond,cpos,upos = self.prepare_fn([prompt],self.tokenizer,self.nvisual,
            self.bos,self.eos,self.boi,self.eoi,self.pad,self.imgpad,self.maxtext,self.device)
        tokens,positions = torch.cat((cond,uncond)),torch.cat((cpos,upos))
        mask = self.mask_fn(2,self.maxlen,positions,self.device).to(self.dtype)
        return dict(text_tokens=tokens,attention_mask=mask,modality_positions=positions,
                    max_seq_len=self.maxlen,guidance_scale=self.cfg,
                    output_hidden_states=True)

    def noise(self, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        x = torch.randn((1,self.channels,self.h*self.patch,self.w*self.patch),
                        generator=generator,device=self.device,dtype=self.dtype)
        return torch.cat((x,x))

    def velocity(self, x, t, kwargs):
        tvec = torch.ones(x.size(0),device=self.device)*t
        return self.drift(x,tvec,self.generation_body,**kwargs).float()

    def from_prefix(self, prefix, x, t, kwargs, repair=None):
        """Saved prefix is AFTER original block 7; only run blocks 8 and 9."""
        hidden = prefix.to(self.device)
        time = self.model.time_embed(torch.ones(x.size(0),device=self.device)*t,self.dtype)
        positions = kwargs['modality_positions']
        block_kwargs = dict(attention_mask=kwargs['attention_mask'],
            position_ids=torch.arange(hidden.shape[1],device=self.device).unsqueeze(0),
            modality_positions=positions)
        if repair is not None:
            hidden = repair(hidden_states=hidden,adaln_input=time,**block_kwargs)[0]
        for block in self.original[8:]:
            hidden = block(hidden_states=hidden,adaln_input=time,**block_kwargs)[0]
        patches = self.model.diffusion_head_b(hidden,time,positions)
        outputs=[]
        for i,(offset,length) in enumerate(positions[:,0].tolist()):
            begin=offset+int(self.model.config.add_time_embeds)
            p = self.model.unpatchify(patches[i:i+1,begin:offset+length],
                                      x.shape[-2]//self.patch,x.shape[-1]//self.patch)
            outputs.append(p.transpose(1,2).reshape(1,*x.shape[1:]))
        raw=torch.cat(outputs)
        cond,uncond=raw.chunk(2)
        guided=uncond+self.cfg*(cond-uncond)
        return torch.cat((guided,guided)).float()

    def decode(self, latent):
        from PIL import Image
        from utils import denorm
        with torch.no_grad(),torch.autocast('cuda',dtype=self.dtype):
            image=self.vae.batch_decode(latent[:1].unsqueeze(2)).squeeze(2)
        return Image.fromarray(denorm(image)[0])

    def metadata(self):
        import subprocess
        result=subprocess.run(['git','-C',str(self.source),'rev-parse','HEAD'],
                              capture_output=True,text=True)
        return {'git_commit':result.stdout.strip(),'checkpoint':'show-o2-1.5B-local',
            'base_model_hash':'a596cbc305c1df987c125d4f218e78f39b681621904cccfb2a3bf0ca0327f92c',
            'sampler':'released shifted Euler','time_grid':self.grid.detach().cpu().tolist(),
            'CFG':self.cfg,'nodes':self.steps,'first_visit_config':'frozen released 10 blocks',
            'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'model_unique_parameters':sum(p.numel() for p in self.model.parameters())}


def replay(runtime, state, repair=None, extra_subdivisions=1, prompt_override=None,
           window=4, checkpoint_rollout=False, extra_steps=1,velocity_modifier=None):
    """Replay a saved float32 state. No re-noising, rounding or seed regeneration."""
    from torch import nn
    from torch.utils.checkpoint import checkpoint
    if checkpoint_rollout and velocity_modifier is not None:
        raise ValueError('Regional oracle is an inference-only teacher, not a training rollout')
    x=state['latent'].to(runtime.device).clone()
    grid=state['time_grid'].to(runtime.device)
    if not torch.equal(grid,runtime.grid):
        raise RuntimeError('Saved and current solver grids differ')
    if state['CFG']!=runtime.cfg or state['prefix_cache_version']!='after-original-block-7-v1':
        raise RuntimeError('Captured CFG or prefix version differs')
    if tensor_hash(state['prefix_cache'])!=state['prefix_cache_hash']:
        raise RuntimeError('Captured prefix corrupted')
    if state['runtime']['base_model_hash']!='a596cbc305c1df987c125d4f218e78f39b681621904cccfb2a3bf0ca0327f92c':
        raise RuntimeError('Captured base-model hash differs')
    kwargs=runtime.inputs(prompt_override or state['prompt'])
    start=state['flow_index']
    runtime.model.diffusion_head_a=nn.ModuleList(runtime.original)
    calls,blocks,revisits=0,0,0
    for i in range(start,len(grid)-1):
        parts=extra_subdivisions if i<start+extra_steps else 1
        if parts != 1 and repair is not None:
            raise ValueError('Extra ODE and recurrence are distinct controls')
        dt=(grid[i+1]-grid[i])/parts
        for j in range(parts):
            t=grid[i]+j*dt
            active=repair is not None and i<start+window
            if active:
                repair.set_time(float(t))
                runtime.model.diffusion_head_a=nn.ModuleList(
                    runtime.original[:8]+[repair]+runtime.original[8:])
            else:
                runtime.model.diffusion_head_a=nn.ModuleList(runtime.original)
            if i==start and j==0 and prompt_override is None:
                def predict(z, now=t, options=kwargs, active_repair=repair if active else None):
                    return runtime.from_prefix(state['prefix_cache'],z,now,options,active_repair)
                blocks+=2
            else:
                def predict(z, now=t, options=kwargs):
                    return runtime.velocity(z,now,options)
                blocks+=10
            # Bind action/time/layers per step: checkpoint recomputation must not
            # read mutable repair contexts from a later flow step.
            if checkpoint_rollout:
                layers=list(runtime.model.diffusion_head_a)
                condition=repair.condition.clone() if active else None
                context=(repair.mode,repair.generic_gain,
                         repair.probe_beta.clone() if repair.probe_beta is not None else None) if active else None
                base_fn=predict
                def bound(z, fn=base_fn, layers=layers, condition=condition,
                          context=context, now=float(t)):
                    runtime.model.diffusion_head_a=nn.ModuleList(layers)
                    if condition is not None:
                        repair.condition=condition
                        repair.mode,repair.generic_gain,repair.probe_beta=context
                    return fn(z)
                velocity=checkpoint(bound,x,use_reentrant=False)
            else:
                velocity=predict(x)
            calls+=1
            if velocity_modifier is not None and i<start+window:
                velocity,added_calls,added_blocks=velocity_modifier(runtime,x,t,velocity)
                calls+=added_calls;blocks+=added_blocks
            if active:
                blocks+=repair.executed_core_calls
                revisits+=repair.executed_core_calls
            x=x+dt*velocity
    runtime.model.diffusion_head_a=nn.ModuleList(runtime.original)
    return x,{'network_velocity_calls':calls,'visual_block_calls':blocks,
              'shared_revisit_calls':revisits,'mmu_checks':0,
              'FLOPs':None}
