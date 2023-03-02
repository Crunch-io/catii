import os

import numpy
from setuptools import Extension, find_packages, setup

numpy_include = numpy.get_include()

ext_modules = [
    Extension(
        name="catii.set_operations",
        sources=["src/catii/set_operations.pyx"],
        include_dirs=[numpy_include],
    ),
]

# For stick-in-the-mud devs:
if os.environ.get("CYTHONIZE_SETUP_PY", False):
    import Cython
    import Cython.Compiler.Options
    from Cython.Build import cythonize

    Cython.Compiler.Options.annotate = True

    ext_modules = cythonize(ext_modules)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="catii",
    version="1.0.0a4",
    author="Robert Brewer, Slobodan Ilić",
    author_email="dev@crunch.io",
    description="A library for N-dimensional categorical data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Crunch-io/catii",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=["numpy>=1.15.2", "Cython>=0.20"],
    tests_require=["pytest"],
    ext_modules=ext_modules,
    entry_points={},
)
