from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


DET_SIZE = (256, 256)
DET_THRESH = 0.2


class ArcFaceExtractor:
    def __init__(self, models_root: Path, device: str) -> None:
        from insightface.app import FaceAnalysis

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"]
        ctx_id = int(device.split(":", 1)[1]) if ":" in device else (0 if device == "cuda" else -1)
        self.device = device
        self.app = FaceAnalysis(name="antelopev2", root=str(models_root), providers=providers, allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=ctx_id, det_thresh=DET_THRESH, det_size=DET_SIZE)

    def __call__(self, rgb: np.ndarray) -> np.ndarray | None:
        faces = self.app.get(rgb[:, :, ::-1].copy())
        if not faces:
            return None
        height, width = rgb.shape[:2]
        diagonal = (height**2 + width**2) ** 0.5

        def score(face) -> float:
            x0, y0, x1, y1 = face["bbox"]
            offset = (((x0 + x1) / 2 - width / 2) ** 2 + ((y0 + y1) / 2 - height / 2) ** 2) ** 0.5 / diagonal
            return (x1 - x0) * (y1 - y0) * (1.0 - offset) ** 2

        embedding = max(faces, key=score)["embedding"]
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        embedding = torch.tensor(embedding, dtype=dtype, device=self.device)
        return (embedding / torch.linalg.vector_norm(embedding)).float().cpu().numpy()
