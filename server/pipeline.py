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
            red_frame = np.zeros((512, 512, 3), dtype=np.uint8)
            red_frame[:, :, 0] = 255
            
            output_frame = av.VideoFrame.from_ndarray(red_frame, format='rgb24')
            output_frame.pts = self._frame_count
            output_frame.time_base = fractions.Fraction(1, 90000)
            output_frame.key_frame = True
            
            # Add forced codec parameters
            output_frame._codec_context = {
                'profile': 'baseline',
                'level': '3.1',
                'pix_fmt': 'yuv420p',
                'width': 512,
                'height': 512
            }
            
            self._frame_count += 1
            print(f"Generated red frame {self._frame_count}")
            return output_frame
            
        except Exception as e:
            print(f"Frame generation failed: {str(e)}")
            error_frame = np.zeros((512, 512, 3), dtype=np.uint8)
            error_frame[:, :, 1] = 255
            return av.VideoFrame.from_ndarray(error_frame, format='rgb24')

    async def warm(self):
        pass  # No warmup needed for static frames

    async def get_nodes_info(self) -> Dict[str, Any]:
        return {}  # Return empty dict for node info

    async def _process_frame(self, frame: av.VideoFrame):
        # Add timestamp continuity check
        if not hasattr(self, '_last_pts'):
            self._last_pts = -1
        
        if frame.pts <= self._last_pts:
            print(f"Invalid PTS: {frame.pts} <= {self._last_pts}")
            frame.pts = self._last_pts + 1
        
        self._last_pts = frame.pts
        return await self._original_process_frame(frame)