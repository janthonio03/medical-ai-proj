"""Separate padding and ROI weights without choosing an attention weighting operator.

Only an explicitly named no-suppression control can currently execute. A future
weighted consumer must define and validate its mathematical operator separately.
"""
from dataclasses import dataclass

import torch


def binary_positions(padding_mask):
    if padding_mask.ndim != 2 or not torch.all((padding_mask == 0) | (padding_mask == 1)):
        raise ValueError('Position IDs require a binary 2D padding mask')
    positions = padding_mask.long().cumsum(-1) - 1
    positions.masked_fill_(padding_mask == 0, 1)  # identical to pinned Phi for real padding
    return positions


@dataclass
class MaskTransport:
    padding_mask: torch.Tensor
    roi_weights: torch.Tensor
    mapping_policy: str

    @classmethod
    def from_official_inputs(cls, inputs, mapping_policy='official_bbox_and_highlight'):
        ids, weights = inputs['input_ids'], inputs['attention_mask']
        if ids.ndim != 2 or ids.shape[0] != 2 or weights.shape != ids.shape:
            raise ValueError('Expected a two-branch official CAST input of equal shape')
        if not torch.equal(ids[0], ids[1]):
            raise ValueError('Both branches must have the same prefix')
        if not torch.all((weights[0] == 0) | (weights[0] == 1)):
            raise ValueError('Conditional mask must be binary')
        if not torch.isfinite(weights).all() or torch.any(weights[1] <= 0):
            raise ValueError('Expected finite positive unconditional ROI weights')
        # The conditional row is the upstream padding definition. Do not derive
        # validity or positions from the weighted unconditional row.
        padding = weights[:1].long().repeat(2, 1).clone()
        if not torch.all(padding[:, -1] == 1):
            raise ValueError('Right-padded input is still unsupported; guard is not bypassed')
        return cls(padding, weights.clone(), mapping_policy)

    def positions(self):
        return binary_positions(self.padding_mask)

    def weights_for_keys(self, key_length):
        """New generated keys are neutral (1) in this proposed transport contract.

        This does not activate weighting; the extension policy is explicit and
        remains a reconstruction choice until an author operator is established.
        """
        if key_length < self.roi_weights.shape[1]:
            raise ValueError('Cache truncation/sliding-window mapping is not defined')
        neutral = self.roi_weights.new_ones((2, key_length - self.roi_weights.shape[1]))
        return torch.cat([self.roi_weights, neutral], dim=1)

    def generation_inputs(self, original, *, mode):
        if mode != 'identity_control':
            raise NotImplementedError('Weighted attention operator is unverified; only identity_control is allowed')
        if original['input_ids'].shape != self.padding_mask.shape:
            raise ValueError('Transport/input shape mismatch')
        result = dict(original)
        result['attention_mask'] = self.padding_mask.clone()
        # Let the unchanged Phi generation method recompute positions from the
        # binary mask at every step, including KV-cache continuation.
        result.pop('position_ids', None)
        return result

    def summary(self):
        return {'mapping_policy': self.mapping_policy, 'padding_values': torch.unique(self.padding_mask).tolist(),
                'position_ids_equal': torch.equal(self.positions()[0], self.positions()[1]),
                'roi_weight_values': torch.unique(self.roi_weights).tolist(),
                'roi_weights_are_applied': False, 'attention_operator': None,
                'is_cast_evaluation': False, 'full_cast_allowed': False}
