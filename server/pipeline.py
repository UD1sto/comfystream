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
        self._frame_count = 0
        print("Initializing minimal pipeline")

    async def __call__(self, frame: av.VideoFrame):
        print(f"\n{'='*50}\nGenerating test frame {self._frame_count}")
        try:
            # Create static red frame (RGB)
            red_frame = np.zeros((512, 512, 3), dtype=np.uint8)
            red_frame[:, :, 0] = 255  # Red channel
            
            # Create video frame with forced codec parameters
            output_frame = av.VideoFrame.from_ndarray(red_frame, format='rgb24')
            output_frame.pts = self._frame_count
            output_frame.time_base = fractions.Fraction(1, 90000)
            
            # Force baseline profile and keyframe
            output_frame.key_frame = True
            output_frame._codec_context = {
                'profile': 'baseline',
                'level': '3.1',
                'pix_fmt': 'yuv420p'
            }
            
            self._frame_count += 1
            print(f"Generated red frame {self._frame_count}")
            return output_frame
            
        except Exception as e:
            print(f"Frame generation failed: {str(e)}")
            return self._generate_error_frame()

    def _generate_error_frame(self):
        error_frame = np.zeros((512, 512, 3), dtype=np.uint8)
        error_frame[:, :, 1] = 255  # Green channel for error
        return av.VideoFrame.from_ndarray(error_frame, format='rgb24')

    def set_prompt(self, prompt: Dict[Any, Any]):
        self.client.set_prompt(prompt)

    async def warm(self):
        print("Warming up pipeline")  # Debug
        # Fix tensor dimensions to match expected CHW format
        frame = torch.randn(1, 3, 512, 512)  # Changed from HWC to CHW
        
        for i in range(WARMUP_RUNS):
            print(f"Warmup run {i+1}/{WARMUP_RUNS}")
            try:
                await self.predict(frame)
            except Exception as e:
                print(f"Warmup error: {str(e)}")

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