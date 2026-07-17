# Archived demo-mode experiment (2026-05-03)

This document previously described a local MiewID index over HappyWhale and FinID-20 images as a
deployable V1. That conclusion has been retired.

The historical metrics and local artifacts may remain useful for research comparison, but none of
the following establishes commercial-production rights:

- a model checkpoint being downloadable;
- Kaggle competition access or a dataset metadata license by itself;
- local possession of images, embeddings, derivatives, or an index;
- an offline accuracy result.

MiewID requires third-party remote code and has no recorded production license in this project.
The FinID-20 materials reviewed by the project contain conflicting usage signals, and the
HappyWhale images have no project-specific production grant. The old `file://` demo-manifest path is
also intentionally rejected by the production network boundary.

Do not deploy, rebuild, or expose the historical demo index. The authoritative replacement is
[`identifier-service.md`](identifier-service.md): pinned DINOv2 weights plus a written, per-source
commercial-production attestation before any reference image is fetched or any index becomes ready.
