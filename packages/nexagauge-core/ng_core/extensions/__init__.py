"""User-facing extension points for nexa-gauge.

Each extension type owns its own registry module (transforms today;
prompts, metrics, rubrics in the future). The shared :func:`load_extension_file`
loader is dataset-agnostic — it imports any Python file so its
``@register_*`` decorators execute as a side effect.
"""

from ng_core.extensions._loader import load_extension_file
from ng_core.extensions.transforms import (
    TransformFn,
    get_transform,
    list_transforms,
    register_transform,
)

__all__ = [
    "TransformFn",
    "get_transform",
    "list_transforms",
    "load_extension_file",
    "register_transform",
]
