"""Trainable embedding model utilities for the public-orca pipeline."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from fluke_model.orca_data import OrcaManifestRow


class OrcaImageDataset(Dataset):
    """Image dataset backed by an OrcaManifestRow list."""

    def __init__(self, rows: list[OrcaManifestRow], image_size: int = 224, train: bool = True):
        self.rows = rows
        self.label_to_idx = {label: i for i, label in enumerate(sorted({r.individual_id for r in rows}))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        augments: list[transforms.Transform] = [
            transforms.Resize((image_size, image_size)),
        ]
        if train:
            augments.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08),
                ]
            )
        augments.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.transform = transforms.Compose(augments)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        row = self.rows[idx]
        image = Image.open(row.path).convert("RGB")
        label = self.label_to_idx[row.individual_id]
        return self.transform(image), label, row.path


class BalancedBatchSampler(Sampler[list[int]]):
    """Sample P identities and K images per identity for metric learning."""

    def __init__(
        self,
        labels: list[int],
        *,
        identities_per_batch: int = 4,
        images_per_identity: int = 2,
        seed: int = 42,
    ):
        if identities_per_batch < 2:
            raise ValueError("identities_per_batch must be >= 2")
        if images_per_identity < 2:
            raise ValueError("images_per_identity must be >= 2")
        self.labels = labels
        self.identities_per_batch = identities_per_batch
        self.images_per_identity = images_per_identity
        self.seed = seed
        self.by_label: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            self.by_label[label].append(idx)
        self.eligible_labels = sorted(
            label for label, idxs in self.by_label.items() if len(idxs) >= images_per_identity
        )
        if len(self.eligible_labels) < identities_per_batch:
            raise ValueError(
                "not enough identities with repeated images for balanced batches: "
                f"need {identities_per_batch}, got {len(self.eligible_labels)}"
            )

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        labels = list(self.eligible_labels)
        rng.shuffle(labels)
        batches = max(1, len(labels) // self.identities_per_batch)
        for batch_no in range(batches):
            start = batch_no * self.identities_per_batch
            chosen = labels[start : start + self.identities_per_batch]
            if len(chosen) < self.identities_per_batch:
                chosen = rng.sample(self.eligible_labels, self.identities_per_batch)
            batch: list[int] = []
            for label in chosen:
                batch.extend(rng.sample(self.by_label[label], self.images_per_identity))
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return max(1, math.floor(len(self.eligible_labels) / self.identities_per_batch))


class EmbedderNet(nn.Module):
    """Timm backbone plus a small projection head that returns L2-normalized embeddings."""

    def __init__(self, backbone: str = "resnet50", embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"timm is required for trainable embedders: {e}") from e

        self.backbone_name = backbone
        self.embed_dim = embed_dim
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        in_features = int(self.backbone.num_features)
        self.projection = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.projection(features)
        return F.normalize(embeddings, dim=-1)


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = 0.2,
) -> torch.Tensor:
    """Batch-hard triplet loss using cosine distance over normalized embeddings."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D")
    if labels.ndim != 1:
        raise ValueError("labels must be 1D")
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings and labels batch sizes differ")

    sim = embeddings @ embeddings.T
    dist = 1.0 - sim
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    positive_mask = same & ~eye
    negative_mask = ~same

    losses: list[torch.Tensor] = []
    for i in range(labels.shape[0]):
        positives = dist[i][positive_mask[i]]
        negatives = dist[i][negative_mask[i]]
        if positives.numel() == 0 or negatives.numel() == 0:
            continue
        hardest_positive = positives.max()
        hardest_negative = negatives.min()
        losses.append(F.relu(hardest_positive - hardest_negative + margin))
    if not losses:
        return embeddings.sum() * 0.0
    return torch.stack(losses).mean()


@dataclass(frozen=True)
class CheckpointMetadata:
    backbone: str
    embed_dim: int
    image_size: int
    source_dataset: str
    split_seed: int
    num_train_images: int
    num_train_individuals: int


def save_checkpoint(
    path: str | Path,
    model: EmbedderNet,
    metadata: CheckpointMetadata,
    *,
    epoch: int,
    metrics: dict,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "metadata": metadata.__dict__,
            "epoch": epoch,
            "metrics": metrics,
        },
        out,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[EmbedderNet, dict]:
    payload = torch.load(path, map_location=device)
    metadata = payload["metadata"]
    model = EmbedderNet(
        backbone=metadata["backbone"],
        embed_dim=int(metadata["embed_dim"]),
        pretrained=False,
    )
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, metadata


@torch.no_grad()
def embed_rows(
    model: EmbedderNet,
    rows: list[OrcaManifestRow],
    *,
    image_size: int,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    dataset = OrcaImageDataset(rows, image_size=image_size, train=False)
    vectors: list[np.ndarray] = []
    for start in range(0, len(dataset), batch_size):
        chunk = [dataset[i][0] for i in range(start, min(start + batch_size, len(dataset)))]
        batch = torch.stack(chunk).to(device)
        emb = model(batch).cpu().numpy().astype(np.float32)
        vectors.append(emb)
    if not vectors:
        return np.zeros((0, model.embed_dim), dtype=np.float32)
    return np.concatenate(vectors, axis=0)
