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
        print("Initializing pipeline")  # Debug
        self.client = ComfyStreamClient(**kwargs)
        self._frame_count = 0
        self._last_pts = -1

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

    async def __call__(self, frame: av.VideoFrame):
        print(f"\n{'='*50}\nPipeline called with frame: {frame}")  # Debug
        try:
            # Input validation and conversion
            if frame is None:
                print("Received null frame, generating default")
                tensor = await self.client.generate_default_frame()
            else:
                print(f"Processing frame: {frame.width}x{frame.height} format={frame.format.name}")
                tensor = self.preprocess(frame)
            
            print(f"Pre-process tensor: shape={tensor.shape} range=[{tensor.min():.2f}, {tensor.max():.2f}]")
            
            # Process through workflow/prediction
            processed = await self.predict(tensor)
            if processed is None:
                print("WARNING: predict returned None, generating error frame")
                return self._generate_error_frame()
                
            print(f"Post-predict tensor: shape={processed.shape} range=[{processed.min():.2f}, {processed.max():.2f}]")
            
            # Convert back to video frame
            output_frame = self.postprocess(processed)
            
            # Set required metadata
            output_frame.pts = self._frame_count
            output_frame.time_base = fractions.Fraction(1, 90000)
            self._frame_count += 1
            
            # Force keyframe periodically
            if self._frame_count % 30 == 0:
                output_frame.key_frame = True
            
            print(f"Generated output frame {self._frame_count}: {output_frame}")
            return output_frame
            
        except Exception as e:
            print(f"Pipeline error: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._generate_error_frame()

    def preprocess(self, frame: av.VideoFrame) -> torch.Tensor:
        try:
            # Convert to RGB numpy array first
            frame_np = frame.to_ndarray(format="rgb24")
            print(f"Numpy frame: shape={frame_np.shape} range=[{frame_np.min()}, {frame_np.max()}]")
            
            # Normalize to [0,1] and convert to tensor
            tensor = torch.from_numpy(frame_np).float() / 255.0
            
            # Add batch dimension and convert HWC -> CHW
            tensor = tensor.unsqueeze(0).permute(0, 3, 1, 2)
            print(f"Preprocessed tensor: shape={tensor.shape} range=[{tensor.min():.2f}, {tensor.max():.2f}]")
            return tensor
        except Exception as e:
            print(f"Preprocess error: {str(e)}")
            raise

    async def predict(self, frame: torch.Tensor) -> torch.Tensor:
        return await self.client.queue_prompt(frame)

    def postprocess(self, tensor: torch.Tensor) -> av.VideoFrame:
        try:
            print(f"Postprocessing tensor: shape={tensor.shape}")
            # Convert CHW -> HWC
            frame = tensor.squeeze(0).permute(1, 2, 0)
            
            # Denormalize from [-1,1] to [0,255]
            frame = (frame * 127.5 + 127.5).clamp(0, 255).byte()
            
            # Debug: Save every 30th frame
            if self._frame_count % 30 == 0:
                debug_frame = frame.cpu().numpy()
                debug_path = f'debug_frame_{self._frame_count}.png'
                cv2.imwrite(debug_path, cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))
                print(f"Saved debug frame to {debug_path}")
            
            # Convert to VideoFrame
            output = av.VideoFrame.from_ndarray(frame.cpu().numpy(), format='rgb24')
            print(f"Created VideoFrame: {output}")
            return output
        except Exception as e:
            print(f"Postprocess error: {str(e)}")
            raise

    def _generate_error_frame(self) -> av.VideoFrame:
        try:
            # Generate a red frame to indicate error
            frame = np.zeros((512, 512, 3), dtype=np.uint8)
            frame[:,:,0] = 255  # Red channel
            output = av.VideoFrame.from_ndarray(frame, format='rgb24')
            output.pts = self._frame_count
            output.time_base = fractions.Fraction(1, 90000)
            self._frame_count += 1
            print("Generated error frame")
            return output
        except Exception as e:
            print(f"Error frame generation failed: {str(e)}")
            raise

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