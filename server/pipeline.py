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
        self._frame_count = 0  # For debugging

    def set_prompt(self, prompt: Dict[Any, Any]):
        self.client.set_prompt(prompt)

    async def warm(self):
        frame = torch.randn(1, 512, 512, 3)

        for _ in range(WARMUP_RUNS):
            await self.predict(frame)

    async def __call__(self, frame: av.VideoFrame):
        try:
            # Input validation
            if frame is None:
                print("Received null frame, generating default")
                tensor = await self.client.generate_default_frame()
            else:
                print(f"Processing frame: {frame.width}x{frame.height} format={frame.format.name}")
                # Convert input frame to tensor
                tensor = self.preprocess(frame)
            
            print(f"Pre-process tensor: shape={tensor.shape} range=[{tensor.min():.2f}, {tensor.max():.2f}]")
            
            # Process through workflow/prediction
            processed = await self.predict(tensor)
            print(f"Post-predict tensor: shape={processed.shape} range=[{processed.min():.2f}, {processed.max():.2f}]")
            
            # Convert back to video frame
            output_frame = self.postprocess(processed)
            
            # Set required metadata
            output_frame.pts = self._frame_count
            output_frame.time_base = fractions.Fraction(1, 90000)
            self._frame_count += 1
            
            return output_frame
            
        except Exception as e:
            print(f"Pipeline error: {str(e)}")
            # Return a colored frame on error
            return self._generate_error_frame()

    def preprocess(self, frame: av.VideoFrame) -> torch.Tensor:
        # Convert to RGB numpy array first
        frame_np = frame.to_ndarray(format="rgb24")
        print(f"Numpy frame: shape={frame_np.shape} range=[{frame_np.min()}, {frame_np.max()}]")
        
        # Normalize to [0,1] and convert to tensor
        tensor = torch.from_numpy(frame_np).float() / 255.0
        
        # Add batch dimension and convert HWC -> CHW
        tensor = tensor.unsqueeze(0).permute(0, 3, 1, 2)
        return tensor

    async def predict(self, frame: torch.Tensor) -> torch.Tensor:
        return await self.client.queue_prompt(frame)

    def postprocess(self, tensor: torch.Tensor) -> av.VideoFrame:
        # Convert CHW -> HWC
        frame = tensor.squeeze(0).permute(1, 2, 0)
        
        # Denormalize from [-1,1] to [0,255]
        frame = (frame * 127.5 + 127.5).clamp(0, 255).byte()
        
        # Debug: Save every 30th frame
        if self._frame_count % 30 == 0:
            debug_frame = frame.cpu().numpy()
            cv2.imwrite(f'debug_frame_{self._frame_count}.png', debug_frame)
            print(f"Saved debug frame {self._frame_count}")
        
        return av.VideoFrame.from_ndarray(frame.cpu().numpy(), format='rgb24')

    def _generate_error_frame(self) -> av.VideoFrame:
        # Generate a red frame to indicate error
        frame = np.zeros((512, 512, 3), dtype=np.uint8)
        frame[:,:,0] = 255  # Red channel
        output = av.VideoFrame.from_ndarray(frame, format='rgb24')
        output.pts = self._frame_count
        output.time_base = fractions.Fraction(1, 90000)
        self._frame_count += 1
        return output

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