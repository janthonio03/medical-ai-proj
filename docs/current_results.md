# Current Experimental Results

## Overall closed-ended performance

| Method | Accuracy | Delta vs Vanilla | Rescued | Harmed |
|---|---:|---:|---:|---:|
| Vanilla | 87.06% | - | - | - |
| PH | 87.39% | +0.33 pp | 5 | 0 |
| VCD | 87.85% | +0.79 pp | 29 | 17 |
| CRG | 87.99% | +0.92 pp | 22 | 8 |
| LoBA-style (Oracle ROI) | 87.46% | +0.40 pp | 6 | 0 |

## Error coverage

The Vanilla model makes 196 errors on the 1,515 closed-ended QA pairs.

- At least one intervention rescues 37 / 196 errors (18.9%).
- All four interventions fail on 159 / 196 errors (81.1%).
- The union of VCD and CRG rescue sets covers all 37 rescued errors.
- Unique rescues: VCD 13, CRG 3, PH 0, LoBA-style 0.

## Confidence finding

All successful rescues occur in the lower half of the Vanilla error-margin distribution.

- Q1 low wrong-confidence errors: PH 5, VCD 16, CRG 19, LoBA-style 6 rescues.
- Q2: PH 0, VCD 13, CRG 3, LoBA-style 0.
- Q3 and Q4: 0 rescues for every method.

PH and LoBA-style are conservative: their rescued cases have median Vanilla wrong margins of approximately 0.047 and 0.039 respectively, and neither harms any Vanilla-correct case. VCD has the broadest coverage but also the largest harm count.

## Exact McNemar tests against Vanilla

| Method | Raw p | Holm-adjusted p |
|---|---:|---:|
| PH | 0.0625 | 0.1250 |
| VCD | 0.1038 | 0.1250 |
| CRG | 0.0161 | 0.0645 |
| LoBA-style | 0.0313 | 0.0938 |

After Holm correction, none of the four comparisons remains significant at alpha=0.05. The main research claim should therefore focus on **case-specific behavioral differences and rescue/harm trade-offs**, rather than a statistically significant global accuracy improvement.
