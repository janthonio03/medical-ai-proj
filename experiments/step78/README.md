# STEP78 comparison input preparation — revision2

Validated on Ariel by user-provided stdout: SELFTEST, COMPARISON_INPUTS_PREPARED and FIXTURE_VERIFY_PASS. Source SHA256 a04b5cdf8fa4d216082f07f02b0854dc959053b974901242e17a1fcb07f2fc30. See [receipt](../../evidence/step78/preparation_receipt.json) and [continuation](../../docs/CONTINUATION.md).

Run from /nas2/data/janthonio03/maip/cast-rebuild using .envs/cast-phi/bin/python. Distributed install path: step78-prepare/prepare.py. CLI modes selftest, prepare, verify. Existing successful preparation need not be rerun; current next action is retrieval of its generated ZIP.

The preparer validates frozen STEP71/model plans,1061 image/candidate records over96caches, source-pinned prefilter K16 and original8balanced canaries. The portable ZIP contains8images/candidate masks/historical ROI/input metadata and needed runtime/config sources. GT labels are separate from pilot_cases.json. No model weights, inference, ROI reselection or GPU allocation.

Important hash distinction: STEP60C strict_roi_sha256 is binary array bytes; STEP71 cases.roi_sha256 is the complete NPY file. Revision2 validates each separately and byte-copies historical ROI files. Selftest and synthetic integration explicitly use different file/array identities and reject corruption. The first buggy preparation is superseded but its failed evidence directories are preserved.

Current verified stage is CPU preparation. Fixture receipt is user terminal evidence; local ZIP inspection and new GPU comparison are pending. Next: paired GT/Draft selection using identical proposals, raw/repaired public spatial baseline versus actual bias, Temporal OFF/ON; up to4GPUs and progress/completion commands supplied with submission.
