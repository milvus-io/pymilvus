"""
Setup script for pymilvus with optional Cython extensions.

This file enables automatic building of Cython extensions during install.
Configuration is primarily in pyproject.toml.
"""
import sys

from setuptools import Extension, setup

# Try to use Cython if available
try:
    from Cython.Build import cythonize

    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    print("Cython not found. Installing without compiled extensions.", file=sys.stderr)
    print("For better performance, install Cython: pip install Cython", file=sys.stderr)


def get_extensions():
    """Build list of extensions to compile."""
    if not USE_CYTHON:
        return []

    extensions = [
        Extension(
            "pymilvus.client._fast_extract",
            sources=["pymilvus/client/_fast_extract.pyx"],
            extra_compile_args=["-O3"],
            language="c",
        )
    ]

    return cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "embedsignature": True,
        },
        # Don't fail build if Cython source is missing
        force=False,
    )


# setuptools will read most config from pyproject.toml
# We only specify ext_modules here
if __name__ == "__main__":
    setup(
        ext_modules=get_extensions(),
    )
