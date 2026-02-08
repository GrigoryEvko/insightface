#!/usr/bin/env python
import os
import io
import glob
import numpy
import re
import sys
import subprocess
import platform
import logging
from setuptools import setup, find_packages

# Cython extensions are optional -- only needed for insightface[all]
HAS_CYTHON = False
try:
    from distutils.core import Extension
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    HAS_CYTHON = True
except ImportError:
    pass


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


pypandoc_enabled = True
try:
    import pypandoc
    print('pandoc enabled')
    long_description = pypandoc.convert_file('README.md', 'rst')
except (IOError, ImportError, ModuleNotFoundError):
    print('WARNING: pandoc not enabled')
    long_description = open('README.md').read()
    pypandoc_enabled = False

VERSION = find_version('insightface', '__init__.py')

requirements = [
    'numpy',
    'onnx',
    'tqdm',
    'requests',
    'Pillow',
    'easydict',
]

extras_require = {
    'rendering': [
        'scipy',
        'matplotlib',
        'scikit-image',
        'albumentations',
    ],
    'all': [
        'scipy',
        'matplotlib',
        'scikit-image',
        'albumentations',
        'cython',
    ],
}

# --- Data files (always included) ---
data_images = list(glob.glob('insightface/data/images/*.jpg'))
data_images += list(glob.glob('insightface/data/images/*.png'))
data_objects = list(glob.glob('insightface/data/objects/*.pkl'))

data_files = [('insightface/data/images', data_images)]
data_files += [('insightface/data/objects', data_objects)]

# --- Cython extension setup (conditional) ---
ext_modules = []
headers_list = []

if HAS_CYTHON:
    extensions = [
        Extension(
            "insightface.thirdparty.face3d.mesh.cython.mesh_core_cython",
            ["insightface/thirdparty/face3d/mesh/cython/mesh_core_cython.pyx",
             "insightface/thirdparty/face3d/mesh/cython/mesh_core.cpp"],
            language='c++',
        ),
    ]

    data_mesh = list(glob.glob('insightface/thirdparty/face3d/mesh/cython/*.h'))
    data_mesh += list(glob.glob('insightface/thirdparty/face3d/mesh/cython/*.c'))
    data_mesh += list(glob.glob('insightface/thirdparty/face3d/mesh/cython/*.py*'))
    data_files += [('insightface/thirdparty/face3d/mesh/cython', data_mesh)]

    ext_modules = cythonize(extensions)
    headers_list = ['insightface/thirdparty/face3d/mesh/cython/mesh_core.h']

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # macOS: check for Homebrew LLVM (needed for OpenMP in Cython extensions)
    if platform.system() == "Darwin":
        logging.info("Detected macOS. Checking if Homebrew, LLVM, and OpenMP are installed...")

        brew_check = subprocess.run(["which", "brew"], capture_output=True, text=True)
        if brew_check.returncode != 0:
            logging.warning("Homebrew is not installed. You may need to manually install dependencies.")
            logging.warning("   Install Homebrew: https://brew.sh/")
            logging.warning("   Then, run: brew install llvm libomp")
            logging.info("Proceeding without setting the compiler.")
        else:
            llvm_check = subprocess.run(["brew", "--prefix", "llvm"], capture_output=True, text=True)
            if llvm_check.returncode != 0:
                logging.warning("LLVM is not installed. This may cause installation issues.")
                logging.warning("   To install, run: brew install llvm libomp")
                logging.warning("   Then, restart the installation process.")
                logging.info("Proceeding without setting the compiler.")
            else:
                llvm_path = subprocess.getoutput("brew --prefix llvm")
                os.environ["CC"] = f"{llvm_path}/bin/clang"
                os.environ["CXX"] = f"{llvm_path}/bin/clang++"
                logging.info(f"Using compiler: {os.environ['CC']}")
else:
    print("NOTE: Cython not found, skipping mesh extension build. "
          "Install with: pip install insightface[all]")

setup(
    # Metadata
    name='insightface',
    version=VERSION,
    author='InsightFace Contributors',
    author_email='contact@insightface.ai',
    url='https://github.com/deepinsight/insightface',
    description='InsightFace Python Library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    # Package info
    packages=find_packages(exclude=('docs', 'tests', 'scripts')),
    data_files=data_files,
    zip_safe=True,
    include_package_data=True,
    entry_points={"console_scripts": ["insightface-cli=insightface.commands.insightface_cli:main"]},
    install_requires=requirements,
    extras_require=extras_require,
    headers=headers_list,
    ext_modules=ext_modules,
    include_dirs=numpy.get_include(),
)

print('pypandoc enabled:', pypandoc_enabled)

