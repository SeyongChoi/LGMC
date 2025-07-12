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
    version='0.1',
    
    packages=["lgmc", "lgmc.dynamics", "lgmc.init", "lgmc.simulator", "lgmc.utils"],

    # Cython extensions
    ext_modules=cythonize(extensions, language_level=3),
    
    # Dependencies
    # install_requires=[
    #     "numpy>=1.21.0",
    #     "tqdm",
    #     "cython",
    #     "pyyaml"
    #     # 필요 시 추가
    # ],

    # CLI entry point
    entry_points={
        'console_scripts': [
            'lgmc = lgmc.main:main',  # lgmc/main.py 또는 main 함수
        ]
    },

    include_package_data=True,
    zip_safe=False,
)
