#!/usr/bin/env python3
"""Boundary/repair regression checks. These do not test GPU model generation."""
from __future__ import annotations
import argparse
import contextlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys

class GuardFacade:
    def forward(self, attention_mask, batch_size=2):
        is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        if is_padding_right:
            raise ValueError('right padding')
        # A repair must not change mask values or weighted cumsum positions.
        return attention_mask.clone(), attention_mask.long().cumsum(-1) - 1

def expect_error(fn, part):
    try:
        fn()
    except (RuntimeError, ValueError) as error:
        assert part in str(error), str(error)
    else:
        raise AssertionError('Expected failure: ' + part)

def run(fixture, source_only=False):
    if source_only:
        # Local test host lacks Transformers. Only a base interface is stubbed;
        # execute the unchanged upstream CFG tensor operations, not generate().
        import types
        for name in ('transformers', 'transformers.generation', 'transformers.generation.logits_process'):
            sys.modules[name] = types.ModuleType(name)
        sys.modules['transformers.generation.logits_process'].LogitsProcessor = object
    import common
    common.FIXTURE = fixture.resolve()
    import runtime as r
    r.FIXTURE = common.FIXTURE
    import torch
    import numpy as np
    common.verify_fixture()
    r.cpu_selftest()
    api = r.modules()
    facade = SimpleNamespace(model=GuardFacade())
    original = GuardFacade.forward
    binary = torch.ones(2, 7)
    weights = binary.clone()
    weights[1, :3] = .01
    weights[1, 3:] = 3
    ref = facade.model.forward(binary)
    expect_error(lambda: facade.model.forward(weights), 'right padding')
    with r.guard_repair(facade):
        repaired = facade.model.forward(binary)
        assert all(torch.equal(a, b) for a, b in zip(ref, repaired))
        observed, positions = facade.model.forward(weights)
        assert torch.equal(observed, weights)
        assert torch.equal(positions, weights.long().cumsum(-1) - 1)
        zero = binary.clone()
        zero[1, -1] = 0
        expect_error(lambda: facade.model.forward(zero), 'right padding')
    assert GuardFacade.forward is original
    try:
        with r.guard_repair(facade):
            raise ValueError('forced cleanup test')
    except ValueError:
        pass
    assert GuardFacade.forward is original

    class FakeModel:
        generation_config = SimpleNamespace(eos_token_id=2)
        config = SimpleNamespace(eos_token_id=2)
        def generate(self, **kwargs):
            ids = kwargs['input_ids']
            return torch.cat([ids, torch.tensor([[7, 8]])], dim=1)

    class FakeProcessor:
        tokenizer = object()
        def __call__(self, **kwargs):
            return {'input_ids': torch.tensor([[1, 2]])}
        def batch_decode(self, *args, **kwargs):
            return ['Draft only']

    native = r.NativeSelectorModel(FakeModel())
    native.generate(input_ids=torch.tensor([[1]]), pixel_values=None, image_sizes=None,
                    max_new_tokens=32, do_sample=False, num_beams=1)
    assert native.drafts[0]['native_generation_eos'] == 2
    expect_error(lambda: native.generate(input_ids=None, answer='GT'), 'Draft generation contract')
    processor = r.InputOnlyProcessor(FakeProcessor())
    expect_error(lambda: processor(text='q', images=[], return_tensors='pt', bbox=[0]),
                 'annotation channel')

    # Exercise real pilot proposal order, native postprocessing, zero candidates,
    # and the private-label read boundary with a deterministic score facade.
    native_prefilter = api.selector.CounterfactualSelector.prefilter
    class SelectionFacade:
        def __init__(self, **kwargs):
            self.model, self.processor = kwargs['model'], kwargs['processor']
        prefilter = native_prefilter
        def draft_answer(self, image, question):
            self.processor(text=question, images=[image], return_tensors='pt')
            self.model.generate(input_ids=torch.tensor([[1]]), pixel_values=None, image_sizes=None,
                                max_new_tokens=32, do_sample=False, num_beams=1)
            return 'Draft only'
        def loglikelihood(self, *args, **kwargs):
            return 0.0
        def score(self, image, question, mask, target):
            return float(mask.sum())
    api.selector.CounterfactualSelector = SelectionFacade
    cases = common.read(fixture / 'pilot_cases.json')
    original_read = r.read
    def reject_private(path):
        assert 'private_evaluation' not in str(path), 'Draft reached private labels'
        return original_read(path)
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        r.read = reject_private
        try:
            for case in cases:
                dest = temp / case['id']
                dest.mkdir()
                roi, receipt = r.select_one(FakeModel(), FakeProcessor(), api, case, 'Draft', dest)
                assert roi.dtype == np.uint8 and receipt['candidate_order'] == case['prefilter_indices']
                assert receipt['GT_target_used_for_this_selection'] is False
                if not case['candidate_count']:
                    assert receipt['selector_skipped'] and not receipt['draft_generation'] and not roi.any()
        finally:
            r.read = original_read
    print('STEP79_OFFLINE_BOUNDARY_PASS guard_restore_native_positions_GT_isolation_8_cases')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--source-only-no-transformers', action='store_true')
    args = parser.parse_args()
    run(args.fixture, args.source_only_no_transformers)
