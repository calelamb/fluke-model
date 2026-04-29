# M-Model-0 Evaluation Summary

First numbers for the Fluke zero-shot photo-ID prototype. See `fluke/docs/specs/m-model-0-prototype.md` for the spec.

| Embedder | Photos | Individuals | Top-1 | Top-3 | Top-5 | MRR | Wall-clock | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| miewid-msv3 | 300 | 50 | 81.0% | 88.0% | 90.0% | 0.855 | 44.0s | ship-V1 |
| dinov2-small | 300 | 50 | 18.3% | 30.7% | 38.0% | 0.290 | 8.6s | retrain |
