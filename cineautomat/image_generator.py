import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
from skimage import color, filters


def generate_keyframe(prompt, style="Wes Anderson style, pastel colors, symmetrical composition"):
    # Load pre-trained Stable Diffusion model
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate image
    full_prompt = f"{prompt}, {style}"
    image = pipe(full_prompt).images[0]

    # Post-process for Wes Anderson aesthetic
    image_np = np.array(image)

    # Apply pastel color filter
    image_hsv = color.rgb2hsv(image_np)
    image_hsv[:, :, 1] = image_hsv[:, :, 1] * 0.6  # Reduce saturation
    image_np = color.hsv2rgb(image_hsv)

    # Apply slight blur for softness
    image_np = filters.gaussian(image_np, sigma=1, channel_axis=-1)

    # Convert back to PIL Image
    image_processed = Image.fromarray((image_np * 255).astype(np.uint8))
    return image_processed


# Example usage
prompt = "A quirky fox in a pastel-colored cafe reading a book"
keyframe = generate_keyframe(prompt)
keyframe.save("keyframe.png")