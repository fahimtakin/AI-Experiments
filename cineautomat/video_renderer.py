from moviepy import ImageSequenceClip
import numpy as np

from animation import generate_animation


def render_video(frames, output_path="output.mp4", fps=24):
    # Convert PIL Images to numpy arrays for moviepy
    frame_arrays = [np.array(frame) for frame in frames]

    # Create video clip
    clip = ImageSequenceClip(frame_arrays, fps=fps)

    # Write to file
    clip.write_videofile(output_path, codec="libx264")

    return output_path


# Example usage
frames = generate_animation([
    "A quirky fox in a pastel-colored cafe reading a book",
    "A quirky fox in a pastel-colored cafe writing a letter",
    "A quirky fox in a pastel-colored cafe looking out the window"
])
video_path = render_video(frames)
print(f"Video saved to {video_path}")