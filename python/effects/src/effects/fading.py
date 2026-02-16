import cv2
import numpy as np
import random
from typing import Tuple, Optional

def apply_gradient_fade(
    img: np.ndarray,
    min_alpha: float = 0.3,
    max_alpha: float = 1.0,
    gradient_type: str = "random"
) -> np.ndarray:
    """
    Applies a gradient fade effect to an image, simulating worn out ink or uneven lighting.
    
    Args:
        img: Input image (numpy array, BGR or BGRA).
        min_alpha: Minimum opacity (0.0 = fully transparent/faded, 1.0 = original).
        max_alpha: Maximum opacity.
        gradient_type: 'linear', 'radial', or 'random'.
        
    Returns:
        Faded image.
    """
    if gradient_type == "random":
        gradient_type = random.choice(["linear", "radial"])
        
    h, w = img.shape[:2]
    
    # Generate the gradient mask (values 0.0 to 1.0)
    if gradient_type == "linear":
        # Random angle for linear gradient
        angle = random.uniform(0, 2 * np.pi)
        cx, cy = w / 2, h / 2
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Rotate coordinates
        dist = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
        
        # Normalize to 0-1 range
        # The max distance from center in a rectangle is the corner
        max_dist = np.sqrt(cx**2 + cy**2)
        mask = (dist + max_dist) / (2 * max_dist)
        
    elif gradient_type == "radial":
        # Random center for radial gradient (can be outside image for partial fade)
        cx = random.uniform(0, w)
        cy = random.uniform(0, h)
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Calculate distance
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Normalize. Max distance is approx diagonal length
        max_dist = np.sqrt(w**2 + h**2)
        mask = 1.0 - (dist / max_dist)
        # Randomly invert radial mask (fade center vs fade edges)
        if random.random() > 0.5:
            mask = 1.0 - mask
            
    else:
        # Fallback to uniform noise if unknown type
        mask = np.random.uniform(0, 1, (h, w))

    # Clamp and scale mask to alpha range
    mask = np.clip(mask, 0.0, 1.0)
    
    # Adjust random intensity curve (gamma) to make fade non-linear sometimes
    gamma = random.uniform(0.5, 2.0)
    mask = mask ** gamma
    
    # Map 0-1 mask to min_alpha-max_alpha
    alpha_map = min_alpha + mask * (max_alpha - min_alpha)
    
    # Apply to image
    result = img.copy()
    
    # Check if image has alpha channel
    if img.shape[2] == 4:
        # Scale existing alpha channel
        result[:, :, 3] = (result[:, :, 3].astype(np.float32) * alpha_map).astype(np.uint8)
    else:
        # If RGB/BGR, blend towards white (assuming white background paper)
        # Fade = 1.0 means original pixel
        # Fade = 0.0 means white pixel
        
        white = np.ones_like(result) * 255
        
        # Expand dimensions of alpha_map to match channels
        alpha_3ch = np.dstack([alpha_map] * 3)
        
        # Blend
        blended = result.astype(np.float32) * alpha_3ch + white * (1.0 - alpha_3ch)
        result = np.clip(blended, 0, 255).astype(np.uint8)
        
    return result
