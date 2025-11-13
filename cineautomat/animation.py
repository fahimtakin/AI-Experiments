import numpy as np
from PIL import Image
import cv2

from image_generator import generate_keyframe


def interpolate_frames(frame1, frame2, num_frames):
    frames = []
    frame1_np = np.array(frame1)
    frame2_np = np.array(frame2)

    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        interpolated = (1 - alpha) * frame1_np + alpha * frame2_np
        frames.append(Image.fromarray(interpolated.astype(np.uint8)))

    return frames


def generate_animation(prompts, fps=24, duration=10):
    # Generate keyframes
    keyframes = [generate_keyframe(prompt) for prompt in prompts]

    # Calculate frames per keyframe transition
    total_frames = fps * duration
    frames_per_transition = total_frames // (len(keyframes) - 1)

    # Interpolate between keyframes
    all_frames = []
    for i in range(len(keyframes) - 1):
        interpolated = interpolate_frames(keyframes[i], keyframes[i + 1], frames_per_transition)
        all_frames.extend(interpolated)

    return all_frames


# Example usage
prompts = [
    "A quirky fox in a pastel-colored cafe reading a book",
    "A quirky fox in a pastel-colored cafe writing a letter",
    "A quirky fox in a pastel-colored cafe looking out the window"
]
frames = generate_animation(prompts)
for i, frame in enumerate(frames):
    frame.save(f"frame_{i:03d}.png")