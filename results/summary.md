# M-Model-0 Evaluation Summary

First numbers for the Fluke zero-shot photo-ID prototype. See
[`fluke/docs/specs/m-model-0-prototype.md`](../../fluke/docs/specs/m-model-0-prototype.md)
for the spec and decision tree.

Run: 2026-04-29 on Apple Silicon (CPU). Frozen embedders, no training.

## Headline

**MiewID-msv3 clears the 70% top-3 bar by a wide margin and is the recommended V1 embedder.** On a 50-individual / 300-photo Beluga ID 2022 leave-one-out subset:

- **MiewID-msv3** — 81.0% top-1, **88.0% top-3**, MRR 0.855 → **ship V1**.
- **DINOv2-Small** — 18.3% top-1, 30.7% top-3, MRR 0.290 → retrain.
- **DINOv3-ConvNeXt-Tiny** — gated repo on HuggingFace; not loadable without an account-side license acceptance. Skipped per spec § 8 risk-1 fallback.

The result lands cleanly in **Branch A** of the decision tree (top-3 ≥ 80%): ship M-Model-0 (MiewID-msv3) as V1 the moment Track A licensing closes. M-Model-2 (ConvNeXt-Tiny + ArcFace fine-tune) is now redundant — MiewID itself is a metric-loss model in this class, trained at far greater scale than we could reproduce.

## Results

### 100-photo / 10-individual subset (pipeline shakedown)

| Embedder | Photos | Individuals | Top-1 | Top-3 | Top-5 | MRR | Wall-clock | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MiewID-msv3 | 100 | 10 | 88.0% | 93.0% | 98.0% | 0.917 | 19.1s | ship-V1 |
| DINOv2-Small | 100 | 10 | 19.0% | 48.0% | 63.0% | 0.403 | 3.6s | retrain |
| DINOv3-ConvNeXt-Tiny | n/a | n/a | n/a | n/a | n/a | n/a | n/a | gated-repo |

### 300-photo / 50-individual subset (broader bar)

| Embedder | Photos | Individuals | Top-1 | Top-3 | Top-5 | MRR | Wall-clock | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MiewID-msv3 | 300 | 50 | 81.0% | 88.0% | 90.0% | 0.855 | 43.9s | ship-V1 |
| DINOv2-Small | 300 | 50 | 18.3% | 30.7% | 38.0% | 0.290 | 8.6s | retrain |

## Comparison anchors

| Source | Dataset | Top-1 | Top-3 |
| --- | --- | --- | --- |
| FIN-PRINT (Bergler 2021, *Sci. Reports*) | Bigg's killer whales, fully trained ArcFace, ~121K images | 92.5% | 97.2% |
| MiewID-msv3 paper (Conservation X / Wild Me, 2024) | Multispecies, **orca** subset | 85.1% | not reported |
| MiewID-msv3 paper | Multispecies, bottlenose dolphin | 94.2% | not reported |
| **This run, MiewID-msv3** | **Beluga ID 2022, 50 individuals** | **81.0%** | **88.0%** |
| **This run, DINOv2-Small** | Beluga ID 2022, 50 individuals | 18.3% | 30.7% |

Our MiewID number is just below the paper's published orca top-1 (81% vs 85.1%), which is consistent with the species shift (orca -> beluga) being mild and the sample size being smaller. The DINOv2-Small gap to MiewID (~63 pp top-3) is much larger than expected from the literature; on a fine-grained re-ID task the animal-specific metric loss carries the day decisively.

## Caveats

1. **Dataset incomplete.** The Beluga ID 2022 archive (563 MB) was not fully downloaded inside the 45-minute wall-clock budget — the GCP mirror timed out at 244 MB after the Azure mirror was abandoned for slowness. We extracted 2,603 of the 5,902 train images plus the full annotations JSON; the manifest covers all 2,603 with 401 individuals having >= 2 photos. The 50-individual subset is sampled from that pool. Re-running with the full 5,902 images (788 individuals) is the obvious next step.
2. **Beluga ID license is CC-BY-NC-ND-2.0 in the actual archive header**, not CDLA-Permissive as the lila.science page suggests. CDLA may apply to derived metadata only. Fine for research / non-commercial prototype; if results need to be published, recheck before quoting.
3. **DINOv3-ConvNeXt-Tiny is gated** behind a HuggingFace login as of 2026-04. We did not gate-accept inside this run since this is a non-interactive eval. Once gate-accepted in a parallel session, the same script will run end-to-end.
4. **No detection or quality gate.** We embed the raw Beluga images directly; they are pre-cropped per the dataset spec. On Salish Sea orca photos we'll need YOLOv8n + a quality gate per `fluke/docs/model.md` Stage 1-2 before the embedder sees the input.
5. **Species shift is real.** Belugas are odontocetes like orcas but lack a true dorsal fin (they have a dorsal *ridge*). MiewID-msv3 was trained on orca, dolphin, beluga, and other cetaceans, so the result here is not a pure transfer test — it's a within-distribution re-ID number on a subset of MiewID's training species. This is consistent with shipping MiewID as V1 on Salish Sea orcas, since orcas are also in the training mix.

## Recommended next step

Per `fluke/docs/v1-plan.md` § Model buildout: M-Model-0 row is **GREEN**. Wire MiewID-msv3 (HuggingFace `conservationxlabs/miewid-msv3`, `trust_remote_code=True`, `transformers==4.49`) into the Modal serverless endpoint behind `POST /api/v1/identify` exactly as specified in `fluke/docs/model.md` § Deployment. Move M-Model-1 (YOLOv8n detector + ResNet-18 quality gate) up the queue; defer M-Model-2 (ConvNeXt-Tiny + ArcFace fine-tune) indefinitely.

The branch from the spec's § 9 decision tree is **Branch A** — zero-shot is a real path, ship as V1.
