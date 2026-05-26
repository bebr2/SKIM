from __future__ import annotations

import json
from dataclasses import dataclass

from torch.utils.data import Dataset


@dataclass
class ReconstructionSample:
    document: str
    metadata: dict


class ReconstructionDataset(Dataset):
    def __init__(
        self,
        path: str | None = None,
        samples: list[ReconstructionSample] | None = None,
    ) -> None:
        self.samples: list[ReconstructionSample] = []
        if samples is not None:
            self.samples = list(samples)
        elif path is not None:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self.samples.append(
                        ReconstructionSample(
                            document=row["document"],
                            metadata=row.get("metadata", {}),
                        )
                    )
        else:
            raise ValueError("Either 'path' or 'samples' must be provided")
        # self.samples = self.samples[:600000]

    @classmethod
    def from_samples(cls, samples: list[ReconstructionSample]) -> "ReconstructionDataset":
        return cls(samples=samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ReconstructionSample:
        return self.samples[idx]
