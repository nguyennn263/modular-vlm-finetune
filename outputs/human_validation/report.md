# C4 self-check: F1 bucket vs. semantic judgment

Single-rater (assistant) semantic-plausibility check, n=119 scored (+1 excluded: references disagreed with each other). **Not** the originally-scoped 300-500-sample / 2-annotator / Cohen's-kappa protocol -- no image access, no second rater, no inter-rater statistic. See outputs/human_validation/selfcheck_judgments.json for the method note and every individual judgment + reasoning.

| F1 bucket | n | correct | partial | incorrect | nonsensical | semantic-acceptable (correct+partial) |
|---|---:|---:|---:|---:|---:|---:|
| strong | 45 | 36 (80.0%) | 5 (11.1%) | 3 (6.7%) | 1 (2.2%) | 41 (91.1%) |
| partial | 58 | 7 (12.1%) | 18 (31.0%) | 32 (55.2%) | 1 (1.7%) | 25 (43.1%) |
| weak | 3 | 0 (0.0%) | 0 (0.0%) | 3 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| zero | 13 | 1 (7.7%) | 1 (7.7%) | 10 (76.9%) | 1 (7.7%) | 2 (15.4%) |
| **overall** | **119** | **44 (37.0%)** | **24 (20.2%)** | **48 (40.3%)** | **3 (2.5%)** | **68 (57.1%)** |
