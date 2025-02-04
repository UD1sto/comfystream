import asyncio
import torch
from typing import Dict, Optional

class TensorCache:
    def __init__(self):
        self.cache: Dict[str, torch.Tensor] = {}
        self.lock = asyncio.Lock()
        self.current_frame: Optional[torch.Tensor] = None
        self.inputs = []
        self.outputs = []
    
    async def store_frame(self, tensor: torch.Tensor):
        async with self.lock:
            self.current_frame = tensor.detach().clone()
    
    async def get_frame(self) -> Optional[torch.Tensor]:
        async with self.lock:
            return self.current_frame.clone() if self.current_frame is not None else None

    def count(self):
        return len(self.inputs)
        
    def debug(self):
        print(f"Cache: {len(self.inputs)} in, {len(self.outputs)} out")
        if self.inputs:
            t = self.inputs[0]
            print(f"First tensor: {t.shape} {t.dtype} {t.device}")

# Global cache instance for VTuber mode
vtuber_cache = TensorCache()
