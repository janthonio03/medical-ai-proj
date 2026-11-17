"""Paper Eq.(4), with the explicitly requested first_step_policy=current_cfg.

State is local to one sample/condition/trajectory. Previous means the preceding
full-vocabulary CFG vector, never a selected scalar or the preceding Final.
"""
import math
import torch
from repro.arcd_generation import tensor_hash
from repro.arcd_runtime import load_official


class TemporalState:
    def __init__(self,lineage,vocab_size,enabled=True,w_m=1.,w_p=.5):
        if not math.isfinite(w_m) or not math.isfinite(w_p):raise ValueError('Nonfinite temporal parameter')
        self.lineage=lineage;self.vocab_size=vocab_size;self.enabled=enabled
        self.w_m=w_m;self.w_p=w_p;self.previous=None;self.step=0;self.ended=False
        self.pending=None

    def preview(self,current,step):
        if self.ended or step!=self.step or self.pending is not None:raise ValueError('Stale step, EOS, or uncommitted score')
        if current.ndim!=1 or current.numel()!=self.vocab_size or current.dtype!=torch.float32:
            raise ValueError('Expected aligned full-vocabulary FP32 vector')
        if not torch.isfinite(current).all():raise ValueError('Nonfinite CFG')
        current=current.detach().to('cpu',copy=True)
        if not self.enabled or self.step==0:final=current.clone()
        else:final=self.w_m*current-self.w_p*self.previous
        if not torch.isfinite(final).all():raise ValueError('Nonfinite Final')
        receipt={'step':step,'state_lineage':self.lineage,'first_step_policy':'current_cfg',
            'temporal_enabled':self.enabled,'w_m':self.w_m,'w_p':self.w_p,
            'current_cfg_sha256':tensor_hash(current),'previous_cfg_sha256':tensor_hash(self.previous) if self.previous is not None else None,
            'previous_step':step-1 if self.previous is not None else None,'previous_used':bool(self.enabled and step>0),
            'final_sha256':tensor_hash(final),'dtype':'torch.float32','arithmetic_device':'cpu',
            'vocabulary_size':self.vocab_size,'token_id_alignment':'vector index equals LM head token ID',
            'final_is_normalized_log_probability':False,'greedy_argmax':int(final.argmax())}
        self.pending=(current,receipt['greedy_argmax'])
        return final,receipt

    def commit(self,consumed_token,eos_ids=()):
        if self.pending is None or self.ended:raise ValueError('No pending current CFG or already EOS')
        if not 0<=consumed_token<self.vocab_size:raise ValueError('Invalid consumed token ID')
        # Called only AFTER argmax and the actual consumed token are recorded.
        self.previous=self.pending[0];self.pending=None;self.step+=1
        self.ended=consumed_token in eos_ids


def spatial_cfg(conditional,unconditional,ids,device):
    """Exact same pinned processor and GPU FP32 call as STEP51."""
    api,_=load_official()
    raw=torch.cat([conditional,unconditional],0).to(device)
    result=api['ProbCFGLogitsProcessor'](1.5,use_log=True)(ids.repeat(2,1),raw)
    if result.dtype!=torch.float32 or not torch.equal(result[0],result[1]):raise ValueError('CFG dtype/branch duplication changed')
    value=result[0].detach().to('cpu',copy=True)
    del raw,result
    return value


def original_log_distribution(conditional,device):
    raw=conditional.to(device)
    value=torch.nn.functional.log_softmax(raw,dim=-1)[0].detach().to('cpu',copy=True)
    del raw
    return value
