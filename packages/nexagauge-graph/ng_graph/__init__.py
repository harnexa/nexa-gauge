from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nexa-gauge")
except PackageNotFoundError:
    __version__ = "0+unknown"
