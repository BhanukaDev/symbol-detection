import cv2
import numpy as np

def twirl_distortion(image, center=None, radius=None, angle=0.5):
    """
    Applies a twirl (swirl) distortion to an image.

    Args:
        image (numpy.ndarray): The input image.
        center (tuple): The (x, y) coordinates of the twirl center.
                        If None, defaults to the image center.
        radius (int): The radius of the twirl effect. Pixels outside this
                      radius are not affected. If None, defaults to half
                      the minimum of image width/height.
        angle (float): The maximum rotation angle in radians (e.g., 0.5 for a mild swirl).

    Returns:
        numpy.ndarray: The twirled image.
    """
    rows, cols = image.shape[:2]

    if center is None:
        center = (cols // 2, rows // 2)
    if radius is None:
        radius = min(cols, rows) // 2

    map_x = np.zeros((rows, cols), dtype=np.float32)
    map_y = np.zeros((rows, cols), dtype=np.float32)

    for y in range(rows):
        for x in range(cols):
            # Calculate distance and angle relative to the center
            dx = x - center[0]
            dy = y - center[1]
            distance = np.sqrt(dx*dx + dy*dy)
            theta = np.arctan2(dy, dx)

            if distance < radius:
                # Apply twirl effect
                # The rotation angle decreases with distance from the center
                dist_factor = (radius - distance) / radius
                new_theta = theta + angle * dist_factor * dist_factor

                # Map back to new coordinates
                map_x[y, x] = center[0] + distance * np.cos(new_theta)
                map_y[y, x] = center[1] + distance * np.sin(new_theta)
            else:
                # No distortion outside the radius
                map_x[y, x] = x
                map_y[y, x] = y

    distorted_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    return distorted_image