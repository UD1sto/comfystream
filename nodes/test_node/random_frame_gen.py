import torch

class RandomImageGenerator:
    """
    Generates random 512x512 test frames for testing transitions
    Outputs: (IMAGE) - Generated random test frame
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("BOOLEAN", {"default": False}),  # Dummy trigger to force execution
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_random" 
    CATEGORY = "test_node"

    def generate_random(self, trigger):
        # Create random image with proper dimensions (1 batch, 512x512, 3 channels)
        img = torch.rand((1, 512, 512, 3))  # Add batch dimension and proper channel order
        return (img,)

NODE_CLASS_MAPPINGS = {
    "RandomImageGenerator": RandomImageGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomImageGenerator": "Random Test Frame Generator"
}
