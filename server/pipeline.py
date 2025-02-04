import torch
import av
import numpy as np

from typing import Any, Dict
from comfystream.client import ComfyStreamClient

WARMUP_RUNS = 5



class Pipeline:
    def __init__(self, **kwargs):
        self.client = ComfyStreamClient(**kwargs)

    def set_prompt(self, prompt: Dict[Any, Any]):
        self.client.set_prompt(prompt)

    async def warm(self):
        frame = torch.randn(1, 512, 512, 3)

        for _ in range(WARMUP_RUNS):
            await self.predict(frame)

    def preprocess(self, frame: av.VideoFrame) -> torch.Tensor:
        frame_np = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
        return torch.from_numpy(frame_np).unsqueeze(0)

    async def predict(self, frame: torch.Tensor) -> torch.Tensor:
        return await self.client.queue_prompt(frame)

    def postprocess(self, frame: torch.Tensor) -> av.VideoFrame:
        # Convert CHW -> HWC and denormalize
        frame = frame.squeeze(0).permute(1, 2, 0)  # HWC
        frame = (frame * 127.5 + 127.5).clamp(0, 255).byte()
        return av.VideoFrame.from_ndarray(frame.cpu().numpy())

    async def __call__(self, frame: av.VideoFrame):
        if frame is None:
            frame = self._generate_solid_frame((512, 512))  # Black frame
        
        # Verify input dimensions
        if frame.width != 512 or frame.height != 512:
            frame = frame.reformat(512, 512)
        
        # Add format check
        if frame.format.name != 'rgb24':
            frame = frame.reformat(format='rgb24')
        
        return await self._process_frame(frame)

    async def get_nodes_info(self) -> Dict[str, Any]:
        """Get information about all nodes in the current prompt including metadata."""
        nodes_info = await self.client.get_available_nodes()
        return nodes_info