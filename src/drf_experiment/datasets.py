from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
try:
    from torchvision import datasets, transforms
except ModuleNotFoundError:
    datasets = None
    transforms = None

from .config import DatasetConfig


class SequenceTensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs.float()
        self.targets = targets.long()

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def _split_dataset(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    val_size: int,
) -> tuple[Dataset[tuple[torch.Tensor, torch.Tensor]], Dataset[tuple[torch.Tensor, torch.Tensor]]]:
    if val_size <= 0:
        return dataset, dataset
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))


def _build_sine_frequency(cfg: DatasetConfig) -> tuple[Dataset, Dataset, Dataset]:
    freqs = torch.linspace(1, cfg.num_classes, cfg.num_classes)

    def make_split(size: int) -> SequenceTensorDataset:
        labels = torch.randint(0, cfg.num_classes, (size,))
        t = torch.linspace(0, 1, cfg.sequence_length)
        signals = []
        for label in labels:
            freq = freqs[label]
            phase = torch.rand(1) * 2 * math.pi
            signal = torch.sin(2 * math.pi * freq * t + phase)
            signal += 0.1 * torch.randn_like(signal)
            signals.append(signal[:, None])
        return SequenceTensorDataset(torch.stack(signals), labels)

    return make_split(cfg.train_size), make_split(cfg.val_size), make_split(cfg.test_size)


def _build_chirp(cfg: DatasetConfig) -> tuple[Dataset, Dataset, Dataset]:
    def make_split(size: int) -> SequenceTensorDataset:
        labels = torch.randint(0, cfg.num_classes, (size,))
        t = torch.linspace(0, 1, cfg.sequence_length)
        signals = []
        for label in labels:
            start = 1.0 + label.item()
            end = start + 1.5 + 0.5 * label.item()
            phase = 2 * math.pi * (start * t + 0.5 * (end - start) * t.square())
            signal = torch.sin(phase) + 0.05 * torch.randn_like(t)
            signals.append(signal[:, None])
        return SequenceTensorDataset(torch.stack(signals), labels)

    return make_split(cfg.train_size), make_split(cfg.val_size), make_split(cfg.test_size)


def _build_delayed_xor(cfg: DatasetConfig) -> tuple[Dataset, Dataset, Dataset]:
    delay = max(4, cfg.sequence_length // 8)

    def make_split(size: int) -> SequenceTensorDataset:
        bits = torch.randint(0, 2, (size, cfg.sequence_length, 2)).float()
        targets = ((bits[:, 0, 0].long() ^ bits[:, delay, 1].long()) % cfg.num_classes).long()
        return SequenceTensorDataset(bits, targets)

    return make_split(cfg.train_size), make_split(cfg.val_size), make_split(cfg.test_size)


def _build_adding(cfg: DatasetConfig) -> tuple[Dataset, Dataset, Dataset]:
    classes = max(2, cfg.num_classes)

    def make_split(size: int) -> SequenceTensorDataset:
        values = torch.rand(size, cfg.sequence_length, 1)
        markers = torch.zeros(size, cfg.sequence_length, 1)
        pos = torch.stack([torch.randperm(cfg.sequence_length)[:2] for _ in range(size)])
        for idx in range(size):
            markers[idx, pos[idx, 0], 0] = 1
            markers[idx, pos[idx, 1], 0] = 1
        targets = values[markers.bool()].view(size, 2).sum(dim=1)
        bins = torch.linspace(0, 2, classes + 1)
        labels = torch.bucketize(targets, bins[1:-1])
        return SequenceTensorDataset(torch.cat([values, markers], dim=-1), labels)

    cfg = replace(cfg, input_dim=2)
    return make_split(cfg.train_size), make_split(cfg.val_size), make_split(cfg.test_size)


def _build_burst_suppression(cfg: DatasetConfig) -> tuple[Dataset, Dataset, Dataset]:
    def make_split(size: int) -> SequenceTensorDataset:
        labels = torch.randint(0, cfg.num_classes, (size,))
        xs = torch.zeros(size, cfg.sequence_length, 1)
        for i, label in enumerate(labels):
            if label.item() % 2 == 0:
                for t in range(0, cfg.sequence_length, 8):
                    xs[i, t : min(t + 3, cfg.sequence_length), 0] = 1.0
            else:
                idx = torch.randperm(cfg.sequence_length)[: cfg.sequence_length // 5]
                xs[i, idx, 0] = 1.0
        xs += 0.02 * torch.randn_like(xs)
        return SequenceTensorDataset(xs, labels)

    return make_split(cfg.train_size), make_split(cfg.val_size), make_split(cfg.test_size)


def _flatten_image_sequence(x: torch.Tensor) -> torch.Tensor:
    return x.view(-1, 1)


def _build_mnist(cfg: DatasetConfig, train: bool, permuted: bool) -> Dataset:
    if datasets is None or transforms is None:
        raise ModuleNotFoundError("torchvision is required for MNIST sequence datasets.")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(_flatten_image_sequence)])
    ds = datasets.MNIST(root=cfg.root, train=train, download=True, transform=transform)
    if not permuted:
        return ds

    generator = torch.Generator().manual_seed(0)
    perm = torch.randperm(28 * 28, generator=generator)

    class PermutedDataset(Dataset):
        def __len__(self) -> int:
            return len(ds)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            x, y = ds[index]
            return x[perm], y

    return PermutedDataset()


def _build_smnist(cfg: DatasetConfig) -> tuple[Dataset, Dataset, Dataset]:
    train_ds, val_ds = _split_dataset(_build_mnist(cfg, True, False), cfg.val_size)
    test_ds = _build_mnist(cfg, False, False)
    return train_ds, val_ds, test_ds


def _build_psmnist(cfg: DatasetConfig) -> tuple[Dataset, Dataset, Dataset]:
    train_ds, val_ds = _split_dataset(_build_mnist(cfg, True, True), cfg.val_size)
    test_ds = _build_mnist(cfg, False, True)
    return train_ds, val_ds, test_ds


def _build_seq_cifar10(cfg: DatasetConfig) -> tuple[Dataset, Dataset, Dataset]:
    if datasets is None or transforms is None:
        raise ModuleNotFoundError("torchvision is required for sequential CIFAR-10.")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.permute(1, 2, 0).reshape(-1, 3)),
        ]
    )
    train = datasets.CIFAR10(root=cfg.root, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root=cfg.root, train=False, download=True, transform=transform)
    train_ds, val_ds = _split_dataset(train, cfg.val_size)
    return train_ds, val_ds, test


def _build_shd(cfg: DatasetConfig) -> tuple[Dataset, Dataset, Dataset]:
    try:
        from spikingjelly.datasets.shd import SpikingHeidelbergDigits
    except Exception as exc:
        raise RuntimeError(
            "SHD requires spikingjelly SHD dependencies. Install h5py and related packages in the snn environment."
        ) from exc

    frame_transform = lambda events: torch.as_tensor(events, dtype=torch.float32)
    train_ds = SpikingHeidelbergDigits(root=cfg.root, train=True, data_type="frame", frames_number=cfg.sequence_length, split_by="number", transform=frame_transform)
    test_ds = SpikingHeidelbergDigits(root=cfg.root, train=False, data_type="frame", frames_number=cfg.sequence_length, split_by="number", transform=frame_transform)

    class Adapted(Dataset):
        def __init__(self, base: Dataset):
            self.base = base

        def __len__(self) -> int:
            return len(self.base)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            x, y = self.base[index]
            x = torch.as_tensor(x, dtype=torch.float32)
            x = x.flatten(1)
            return x, torch.tensor(y)

    train_ds, val_ds = _split_dataset(Adapted(train_ds), cfg.val_size)
    return train_ds, val_ds, Adapted(test_ds)


def _build_tensor_folder(cfg: DatasetConfig) -> tuple[Dataset, Dataset, Dataset]:
    base = Path(cfg.root) / cfg.name

    def load_split(split: str) -> SequenceTensorDataset:
        pt = base / f"{split}.pt"
        npz = base / f"{split}.npz"
        if pt.exists():
            payload = torch.load(pt, map_location="cpu")
            return SequenceTensorDataset(payload["inputs"], payload["targets"])
        if npz.exists():
            payload = np.load(npz)
            return SequenceTensorDataset(torch.from_numpy(payload["inputs"]), torch.from_numpy(payload["targets"]))
        raise FileNotFoundError(f"Missing prepared split for {cfg.name}: expected {pt} or {npz}")

    return load_split("train"), load_split("val"), load_split("test")


BUILDERS: dict[str, Callable[[DatasetConfig], tuple[Dataset, Dataset, Dataset]]] = {
    "sine_frequency": _build_sine_frequency,
    "chirp": _build_chirp,
    "delayed_xor": _build_delayed_xor,
    "adding": _build_adding,
    "burst_suppression": _build_burst_suppression,
    "smnist": _build_smnist,
    "psmnist": _build_psmnist,
    "scifar10": _build_seq_cifar10,
    "shd": _build_shd,
    "lra_listops": _build_tensor_folder,
    "lra_text": _build_tensor_folder,
    "lra_retrieval": _build_tensor_folder,
    "lra_image": _build_tensor_folder,
    "lra_pathfinder": _build_tensor_folder,
}

AVAILABLE_DATASETS = tuple(BUILDERS.keys())

DATASET_DEFAULTS: dict[str, dict[str, int]] = {
    "sine_frequency": {"input_dim": 1, "num_classes": 4, "sequence_length": 128},
    "chirp": {"input_dim": 1, "num_classes": 4, "sequence_length": 128},
    "delayed_xor": {"input_dim": 2, "num_classes": 2, "sequence_length": 128},
    "adding": {"input_dim": 2, "num_classes": 4, "sequence_length": 128},
    "burst_suppression": {"input_dim": 1, "num_classes": 2, "sequence_length": 128},
    "smnist": {"input_dim": 1, "num_classes": 10, "sequence_length": 784},
    "psmnist": {"input_dim": 1, "num_classes": 10, "sequence_length": 784},
    "scifar10": {"input_dim": 3, "num_classes": 10, "sequence_length": 1024},
    "shd": {"input_dim": 700, "num_classes": 20, "sequence_length": 250},
    "lra_listops": {"input_dim": 18, "num_classes": 10, "sequence_length": 2048},
    "lra_text": {"input_dim": 256, "num_classes": 2, "sequence_length": 4096},
    "lra_retrieval": {"input_dim": 256, "num_classes": 2, "sequence_length": 4000},
    "lra_image": {"input_dim": 1, "num_classes": 10, "sequence_length": 1024},
    "lra_pathfinder": {"input_dim": 1, "num_classes": 2, "sequence_length": 1024},
}


def build_dataloaders(cfg: DatasetConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    if cfg.name not in BUILDERS:
        raise KeyError(f"Unsupported dataset: {cfg.name}")
    train_ds, val_ds, test_ds = BUILDERS[cfg.name](cfg)
    base_loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory and torch.cuda.is_available(),
    }
    train_loader_kwargs = dict(base_loader_kwargs)
    eval_loader_kwargs = dict(base_loader_kwargs)
    if cfg.num_workers > 0:
        persistent_workers = cfg.persistent_workers
        train_loader_kwargs["persistent_workers"] = persistent_workers
        eval_loader_kwargs["persistent_workers"] = persistent_workers
        if cfg.prefetch_factor > 0:
            train_loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
            eval_loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
    return (
        DataLoader(train_ds, shuffle=True, **train_loader_kwargs),
        DataLoader(val_ds, shuffle=False, **eval_loader_kwargs),
        DataLoader(test_ds, shuffle=False, **eval_loader_kwargs),
    )


def collect_frequency_init_samples(loader: DataLoader, max_batches: int = 4) -> torch.Tensor:
    batches = []
    for index, (x, _) in enumerate(loader):
        batches.append(x)
        if index + 1 >= max_batches:
            break
    return torch.cat(batches, dim=0)


def apply_dataset_defaults(cfg: DatasetConfig) -> DatasetConfig:
    defaults = DATASET_DEFAULTS.get(cfg.name, {})
    for key, value in defaults.items():
        setattr(cfg, key, value)
    return cfg
