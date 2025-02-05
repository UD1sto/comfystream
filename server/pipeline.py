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

    async def run_continuous(self, fps: int = 30):
        """Proper frame rate controller"""
        frame_interval = 1 / fps
        last_frame_time = time.monotonic()

        while True:
            start = time.monotonic()
            yield await self()  # Generate frame
            
            # Precision timing control
            elapsed = time.monotonic() - start
            sleep_duration = max(0, frame_interval - elapsed)
            await asyncio.sleep(sleep_duration)
            
            # Emergency catch for time drift
            if time.monotonic() - last_frame_time > 5 * frame_interval:
                logger.warning("Frame generation lagging behind realtime")
                last_frame_time = time.monotonic()
            
        

    async def __call__(self) -> av.VideoFrame:
        """Generate frames without input, using internal frame generation."""
        generated_tensor = await self.generate()  # Get tensor from auto-prompt
        post_output = self.postprocess(generated_tensor)
        
        # Maintain consistent timing (90kHz clock for video)
        post_output.pts = int(time.time() * 90000)
        post_output.time_base = av.time_base // 1000
        
        return post_output
    
   