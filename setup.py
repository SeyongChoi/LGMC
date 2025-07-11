from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="lgmc.dynamics.kawasaki",
        sources=["lgmc/dynamics/kawasaki.pyx"],
        include_dirs=[np.get_include()],
        language="c",
    )
]

setup(
    name="lgmc",
    packages=["lgmc", "lgmc.dynamics", "lgmc.init", "lgmc.utils"],
    ext_modules=cythonize(extensions, language_level=3),
)
