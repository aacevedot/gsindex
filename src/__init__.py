from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("gsindex")
except PackageNotFoundError:
    # package is not installed
    pass