import torch
import av
import numpy as np
import time
import asyncio
import logging

from typing import Any, Dict
from comfystream.client import ComfyStreamClient

WARMUP_RUNS = 5

logger = logging.getLogger(__name__)



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
        return av.VideoFrame.from_ndarray(
            (frame * 255.0).clamp(0, 255).to(dtype=torch.uint8).squeeze(0).cpu().numpy()
        )

    async def __call__(self, frame: av.VideoFrame) -> av.VideoFrame:
        pre_output = self.preprocess(frame)
        pred_output = await self.predict(pre_output)
        post_output = self.postprocess(pred_output)

        post_output.pts = frame.pts
        post_output.time_base = frame.time_base

        return post_output

    async def get_nodes_info(self) -> Dict[str, Any]:
        """Get information about all nodes in the current prompt including metadata."""
        nodes_info = await self.client.get_available_nodes()
        return nodes_info

class OneWayPipeline(Pipeline):
    def __init__(self, **kwargs):
        self.client = ComfyStreamClient(**kwargs)

        
    def set_prompt(self, prompt: Dict[Any, Any]):
        """Override to use auto-prompt setup"""
        self.client.set_prompt(prompt)

    async def generate(self) -> torch.Tensor:
        """Use auto-generation queue method"""
        return await self.client.queue_prompt_auto()
    
    def postprocess(self, frame: torch.Tensor) -> av.VideoFrame:
        return av.VideoFrame.from_ndarray(
            (frame * 255.0).clamp(0, 255).to(dtype=torch.uint8).squeeze(0).cpu().numpy()
        )

    async def __call__(self) -> av.VideoFrame:
        """Generate frames without input using direct H.264 output from comfyui workflow."""
        # Instead of generating a tensor and then postprocessing, we assume the comfyui workflow
        # directly produces an encoded frame.
        encoded_frame = await self.generate()  # Now returns an av.VideoFrame already in H264 compatible format
        
        # Maintain consistent timing (using a 90kHz clock for video)
        encoded_frame.pts = int(time.time() * 90000)
        encoded_frame.time_base = av.time_base // 1000
        return encoded_frame
    
   