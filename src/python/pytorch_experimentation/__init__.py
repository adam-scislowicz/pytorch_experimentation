"""Top-level package for pytorch_experimentation."""

import atexit

__author__ = """Adam Scislowicz"""
__email__ = "adam.scislowicz@gmail.com"
__version__ = "0.0.1"


def shutdown() -> None:
    """This function is called on module exit."""

    print("pytorch_experimentation module shutdown")


atexit.register(shutdown)

print("pytorch_experimentation module loaded.")
