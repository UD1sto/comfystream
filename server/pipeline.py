import torch
import av
import numpy as np
import fractions
import cv2

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

    def postprocess(self, frame: torch.Tensor):
        # Temporary debug export
        debug_frame = frame.clone().squeeze(0).permute(1,2,0).cpu().numpy()
        cv2.imwrite('debug_frame.png', debug_frame)
        return av.VideoFrame.from_ndarray(debug_frame)

    async def __call__(self, frame: av.VideoFrame):
        # Force baseline profile in SDP answer
        frame.force_keyframe()
        frame.time_base = fractions.Fraction(1, 90000)  # Match WebRTC clock
        return frame

    async def get_nodes_info(self) -> Dict[str, Any]:
        """Get information about all nodes in the current prompt including metadata."""
        nodes_info = await self.client.get_available_nodes()
        return nodes_info

    async def _process_frame(self, frame: av.VideoFrame):
        # Add timestamp continuity check
        if not hasattr(self, '_last_pts'):
            self._last_pts = -1
        
        if frame.pts <= self._last_pts:
            print(f"Invalid PTS: {frame.pts} <= {self._last_pts}")
            frame.pts = self._last_pts + 1
        
        self._last_pts = frame.pts
        return await self._original_process_frame(frame)