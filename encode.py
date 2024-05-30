import numpy as np
import json
import base64

# Convert ndarray to a base64 encoded string
def ndarray_to_base64(array: np.ndarray):
    array_bytes = array.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string


