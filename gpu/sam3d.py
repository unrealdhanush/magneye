"""SAM 3D Objects on a Modal GPU: photo + object mask in, 3D shell out.

Not yet run end to end. Before deploying:
  1. Request access to https://huggingface.co/facebook/sam-3d-objects (gated).
  2. `modal secret create huggingface HF_TOKEN=<token with access>`
  3. `modal deploy gpu/sam3d.py`, then put the printed URL in .env as SAM3D_URL.

SAM 3D Objects needs a >=32 GB GPU and its documented export is a Gaussian splat (PLY).
This endpoint returns a GLB when the pipeline output carries a mesh, and otherwise answers
422 listing the output keys, so the first real run tells us which export to wire up.
"""

from __future__ import annotations

import io

import modal

REPO = "https://github.com/facebookresearch/sam-3d-objects"
CKPT = "/opt/sam3d/checkpoints/hf"

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .run_commands(
        f"git clone --depth 1 {REPO} /opt/sam3d",
        "pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121",
        "cd /opt/sam3d && PIP_EXTRA_INDEX_URL='https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121' pip install -e '.[dev]'",
        "cd /opt/sam3d && PIP_EXTRA_INDEX_URL='https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121' pip install -e '.[p3d]'",
        "cd /opt/sam3d && PIP_FIND_LINKS='https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html' pip install -e '.[inference]'",
        "cd /opt/sam3d && ./patching/hydra",
        "pip install 'huggingface-hub[cli]<1.0' fastapi python-multipart pillow numpy trimesh",
    )
    .run_commands(
        "cd /opt/sam3d && hf download --repo-type model --local-dir checkpoints/hf-download --max-workers 1 facebook/sam-3d-objects"
        " && mv checkpoints/hf-download/checkpoints checkpoints/hf && rm -rf checkpoints/hf-download",
        secrets=[modal.Secret.from_name("huggingface")],
    )
)

app = modal.App("magneye-sam3d", image=image)


@app.cls(gpu="A100-40GB", timeout=900, scaledown_window=300)
class Sam3D:
    @modal.enter()
    def load(self) -> None:
        import sys

        sys.path.append("/opt/sam3d/notebook")
        from inference import Inference  # type: ignore[import-not-found]

        self.inference = Inference(f"{CKPT}/pipeline.yaml", compile=False)

    @modal.fastapi_endpoint(method="POST")
    async def reconstruct(self, request):  # FastAPI Request; imported lazily inside the container
        import numpy as np
        from fastapi import Response
        from PIL import Image

        form = await request.form()
        rgb = np.array(Image.open(io.BytesIO(await form["image"].read())).convert("RGB"))
        mask = np.array(Image.open(io.BytesIO(await form["mask"].read())).convert("L")) > 127
        output = self.inference(rgb, mask, seed=42)

        mesh = next((output[k] for k in ("glb", "mesh") if k in output and output[k] is not None), None)
        if mesh is None:
            return Response(f"no mesh in SAM 3D output; keys: {sorted(output)}", status_code=422)
        buf = io.BytesIO()
        mesh.export(buf, file_type="glb")  # trimesh.Trimesh / Scene
        return Response(buf.getvalue(), media_type="model/gltf-binary")
