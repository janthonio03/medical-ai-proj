"""STEP 51 reconstruction only. No edits to ARCD, Phi forward, RoPE, or CFG.

The CAST-style rule here is a specified reconstruction, not an author's patch.
The TorchFunction observer returns each original operator result unchanged and
copies only the last query row of softmax scores/probabilities to CPU.
"""
import math
import torch
from torch.overrides import TorchFunctionMode

from repro.arcd_generation import tensor_hash


def visual_roi(highlight, input_ids):
    h = highlight.detach().bool().flatten()
    ids = input_ids[0] if input_ids.ndim == 2 else input_ids
    if h.numel() != ids.numel():
        raise ValueError('Highlight/prefix length mismatch')
    return h & (ids.to(h.device) < 0)


def key_highlight(highlight, key_length, device):
    if key_length < highlight.numel():
        raise ValueError('Truncated/sliding prefix unsupported')
    return torch.nn.functional.pad(highlight.to(device=device, dtype=torch.bool),
                                   (0, key_length-highlight.numel()))


def roi_attention_mask(mask, roi, scale, branch):
    """Add log(scale) to valid ROI keys in the eager 4D mask, before QK softmax.

    Binary-causal blocked entries remain bit-for-bit unchanged, including -inf.
    This is the finite-mask realization of S+M+log(s)*H; M remains the same.
    Conditional and empty-ROI paths return the exact original mask object.
    """
    if mask.ndim != 4 or mask.shape[0] != 1 or branch not in (0, 1):
        raise ValueError('Expected single-branch 4D eager attention mask')
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('Suppression scale must be in (0,1]')
    h = key_highlight(roi, mask.shape[-1], mask.device)
    if branch == 0 or scale == 1 or not bool(h.any()):
        return mask
    bias = h.to(mask.dtype).view(1, 1, 1, -1) * math.log(scale)
    # Existing M is zero on valid keys, negative on causal/padding exclusions.
    return torch.where(mask == 0, mask + bias, mask)


class CastSpatialHooks:
    """Independent reconstruction: no embedding changes, no conditional bias."""
    def __init__(self, model, roi, scale=.01):
        self.model = model
        self.roi = roi.detach().bool().flatten()
        self.scale = scale
        self.branch = 0
        self.events = []
        self.handles = []
        self.att_events = {}
        self.peak = 0
        emb = [m for m in model.modules() if type(m).__name__ == 'Phi3ImageEmbedding']
        att = [m for m in model.modules() if type(m).__name__ == 'Phi3Attention']
        if len(emb) != 1 or len(att) != model.config.num_hidden_layers or not att:
            raise ValueError('Require the pinned eager Phi module layout')
        if any(hasattr(m, '_aregion') for m in model.modules()):
            raise ValueError('ARCD hooks/state must be removed before reconstruction')
        self.handles.append(emb[0].register_forward_hook(self.embedding))
        for index, module in enumerate(att):
            self.handles.append(module.register_forward_pre_hook(self.attention(index), with_kwargs=True))
            self.handles.append(module.register_forward_hook(self.after(index)))

    def set(self, branch):
        self.branch = branch
        self.events = []

    def embedding(self, module, args, output):
        self.events.append({'kind':'embedding','branch':self.branch,
            'embedding_untouched':True,
            'prefill':output.shape[1] == self.roi.numel()})
        return None  # PyTorch forward-hook contract: preserve original output.

    def attention(self, index):
        def hook(module, args, kwargs):
            original = kwargs['attention_mask']
            changed = roi_attention_mask(original, self.roi, self.scale, self.branch)
            h = key_highlight(self.roi, original.shape[-1], original.device)
            before = original[:, :, -1, :]
            after = changed[:, :, -1, :]
            eligible = (before == 0) & h.view(1, 1, -1) & (self.branch == 1)
            expected = torch.where(eligible, before + math.log(self.scale), before)
            delta = after - before  # Actual pinned M is finite finfo.min.
            event = {'kind':'attention','layer':index,'branch':self.branch,
                'scale':self.scale,'query_length':original.shape[-2],
                'key_length':original.shape[-1], 'roi_key_count':int(h.sum()),
                'highlight_sha256':tensor_hash(h),
                'generated_keys_highlighted':int(h[self.roi.numel():].sum()),
                'mask_last_query_exact':torch.equal(after, expected),
                'conditional_untouched':torch.equal(original,changed) if self.branch == 0 else None,
                'actual_bias_min':float(delta.min()),'actual_bias_max':float(delta.max()),
                'one_score_tensor_bytes_estimate':module.num_heads*original.shape[-2]*original.shape[-1]*original.element_size()}
            self.events.append(event); self.att_events[index] = event
            if changed.is_cuda:
                self.peak = max(self.peak,torch.cuda.max_memory_allocated(0))
                event['allocated_before_attention'] = torch.cuda.memory_allocated(0)
                torch.cuda.reset_peak_memory_stats(0)
            return args, dict(kwargs,attention_mask=changed)
        return hook

    def after(self,index):
        def hook(module,args,output):
            t = output[0] if isinstance(output,tuple) else output
            if isinstance(t,torch.Tensor) and t.is_cuda:
                peak = torch.cuda.max_memory_allocated(0)
                self.peak = max(self.peak,peak)
                self.att_events[index].update(attention_forward_window_peak=peak,
                    allocated_after_attention=torch.cuda.memory_allocated(0))
        return hook

    def close(self):
        for handle in self.handles: handle.remove()
        self.handles.clear()


class SoftmaxObserver(TorchFunctionMode):
    """Observe the actual eager softmax, without returning altered tensors.

    At most heads*keys scores and probabilities are copied at each layer. The
    full GPU attention matrix is neither cloned nor retained by this observer.
    Reference removes only the explicit bias from the SAME observed score row;
    it is not a comparison to another branch's different QK states.
    """
    def __init__(self, model, roi):
        super().__init__()
        self.roi = roi.detach().bool().flatten()
        self.active = None
        self.records = []
        self.handles = []
        self.bias_highlight = self.roi
        self.bias_log_weight = 0.
        att = [m for m in model.modules() if type(m).__name__ == 'Phi3Attention']
        for i,m in enumerate(att):
            self.handles.append(m.register_forward_pre_hook(self.before(i),with_kwargs=True))
            self.handles.append(m.register_forward_hook(self.after(i)))

    def configure(self, highlight, log_weight):
        self.records = []
        self.bias_highlight = highlight.detach().bool().flatten()
        self.bias_log_weight = float(log_weight)
        self.active = None

    def before(self,index):
        def hook(module,args,kwargs):
            mask = kwargs['attention_mask']
            # Finite minimum excludes causal/padding keys; -4.6 is valid bias.
            valid = (mask[0,0,-1,:] > torch.finfo(mask.dtype).min/2).detach().cpu().clone()
            self.active = (index,valid)
        return hook

    def after(self,index):
        def hook(module,args,output): self.active = None
        return hook

    def __torch_function__(self,func,types,args=(),kwargs=None):
        kwargs = kwargs or {}
        dim=kwargs.get('dim',args[1] if len(args)>1 else None)
        observe = (self.active is not None and func in
            (torch.softmax,torch.Tensor.softmax,torch.nn.functional.softmax)
            and bool(args) and isinstance(args[0],torch.Tensor)
            and args[0].ndim == 4 and dim in (-1,3))
        if observe:
            scores = args[0][:,:,-1,:].detach().float().cpu().clone()
            index,valid = self.active
        result = func(*args,**kwargs)  # The original operation, exactly once.
        if observe:
            probability = result[:,:,-1,:].detach().float().cpu().clone()
            roi = key_highlight(self.roi,scores.shape[-1],'cpu') & valid
            bias_h = key_highlight(self.bias_highlight,scores.shape[-1],'cpu') & valid
            bias = bias_h.to(scores.dtype).view(1,1,-1)*self.bias_log_weight
            reference = torch.softmax(scores-bias,dim=-1)
            actual_roi = probability[...,roi].sum(-1,dtype=torch.float64)
            ref_roi = reference[...,roi].sum(-1,dtype=torch.float64)
            sums = probability[...,valid].sum(-1,dtype=torch.float64)
            excluded = probability[...,~valid]
            eps = torch.finfo(torch.float32).eps
            tolerance = 32*eps  # FP32 softmax and <=4096-key reduction envelope.
            effective = (ref_roi > tolerance) & (ref_roi < 1-tolerance)
            decreases = actual_roi < ref_roi - tolerance
            entry = {'layer':index,'actual_softmax_seen':True,'score_shape':list(args[0].shape),
                'observed_query':'last query only','roi_keys':int(roi.sum()),
                'valid_keys':int(valid.sum()),'normalization_max_error':float((sums-1).abs().max()),
                'masked_probability_max':float(excluded.abs().max()) if excluded.numel() else 0.,
                'probability_tolerance':tolerance,
                'roi_mass_actual_per_head':actual_roi.flatten().tolist(),
                'roi_mass_same_qk_no_bias_reference_per_head':ref_roi.flatten().tolist(),
                'effective_head_count':int(effective.sum()),
                'strict_decrease_head_count':int((decreases & effective).sum()),
                'nonincrease_all_heads':bool((actual_roi <= ref_roi+tolerance).all()),
                'applied_log_weight':self.bias_log_weight,
                'full_attention_matrix_retained':False,
                'observational_only':True}
            self.records.append(entry)
        return result

    def close(self):
        for handle in self.handles: handle.remove()
        self.handles.clear()
