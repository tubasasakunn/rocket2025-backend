# Patch for FastAPI's jsonable_encoder
# This file will be imported in main.py to patch FastAPI's encoding system
# to better handle binary data

import base64
from typing import Any, Dict, List, Union, Optional
from fastapi.encoders import jsonable_encoder as original_jsonable_encoder

def patched_jsonable_encoder(
    obj: Any,
    *args,
    **kwargs
) -> Any:
    """
    Patched version of FastAPI's jsonable_encoder that properly handles binary data
    """
    # Handle bytes directly
    if isinstance(obj, bytes):
        # Convert bytes to base64 string
        return f"bytes:base64:{base64.b64encode(obj).decode('ascii')}"
    
    # Handle dict with potential binary content
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # Recursively encode each value
            if isinstance(v, bytes):
                result[k] = f"bytes:base64:{base64.b64encode(v).decode('ascii')}"
            else:
                result[k] = patched_jsonable_encoder(v, *args, **kwargs)
        return result
    
    # Handle list with potential binary content
    if isinstance(obj, list):
        return [patched_jsonable_encoder(item, *args, **kwargs) for item in obj]
    
    # Use original encoder for other types
    return original_jsonable_encoder(obj, *args, **kwargs)
