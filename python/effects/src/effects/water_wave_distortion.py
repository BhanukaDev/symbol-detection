import cv2
import numpy as np

def water_wave_distortion(image, amplitude=20, frequency=0.05):
    """
    Create water-like wave distortion

    amplitude: strength of the wave (higher = more distortion)
    frequency: how many waves (higher = more ripples)
    """
    rows, cols = image.shape[:2]

    # Create mesh grid
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Apply sine wave distortion
    x_distorted = x + amplitude * np.sin(2 * np.pi * y * frequency)
    y_distorted = y + amplitude * np.sin(2 * np.pi * x * frequency)

    # Remap image
    distorted = cv2.remap(
        image,
        x_distorted.astype(np.float32),
        y_distorted.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return distorted
