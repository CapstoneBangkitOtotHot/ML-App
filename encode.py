import numpy as np
import json
import base64
import secrets
from PIL import Image

# Convert ndarray to a base64 encoded string
def ndarray_to_base64(array: np.ndarray):
    array_bytes = array.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string


def save_ndarray_image(array: np.ndarray, filename: str = None, format: str = "png"):
    """Save ndarray image to a file"""
    img = Image.fromarray(array)

    if filename is None:
        filename = f"{secrets.token_urlsafe(20)}.{format}"

    img.save(f"./converted_images/{filename}")
    img.close()