import asyncio

class H264Cache:
    def __init__(self):
        # For H.264 mode the input list might not be needed,
        # but we include it for symmetry in case it's required.
        self.inputs = []
        # The outputs will store futures that are resolved with H.264 frames.
        self.outputs = []
        self.lock = asyncio.Lock()

h264_cache = H264Cache() 