from importlib.metadata import PackageNotFoundError, version

from ng_core.extensions import (
    get_transform,
    list_transforms,
    load_extension_file,
    register_transform,
)

try:
    __version__ = version("nexa-gauge")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "get_transform",
    "list_transforms",
    "load_extension_file",
    "register_transform",
]
