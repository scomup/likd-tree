#!/usr/bin/env python3
"""
Setup script for likd-tree Python bindings
Run from python/ directory: python setup.py install
Or from root directory: pip install ./python
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import pybind11
from pathlib import Path
import subprocess
import os
import platform


class get_pybind_include:
    """Helper class to determine the pybind11 include path"""

    def __str__(self):
        return pybind11.get_include()


def find_eigen_include():

    result = subprocess.run(['pkg-config', '--cflags-only-I', 'eigen3'], 
                          capture_output=True, text=True, check=False)
    if result.returncode == 0:
        path = result.stdout.strip().replace('-I', '')
        if path and Path(path).exists():
            return path
    return None


# Find Eigen
eigen_include = find_eigen_include()

if not eigen_include:
    print("\n" + "="*70)
    print("ERROR: Could not find Eigen3 headers!")
    print("="*70 + "\n")
    sys.exit(1)

print(f"Found Eigen3 at: {eigen_include}")

# Get the source directory - works both when called from python/ dir and when installed from sdist
# In sdist, src/ is at the same level as python files in likd_tree-1.0.0/src/
src_dir = Path(__file__).parent / 'src'
if not src_dir.exists():
    # Try parent directory (when called from python/ in repo)
    src_dir = Path(__file__).parent.parent / 'src'

print(f"Source directory: {src_dir}")
print(f"likd_tree.hpp exists: {(src_dir / 'likd_tree.hpp').exists()}")

ext_modules = [
    Extension(
        'likd_tree',
        ['likd_tree_py.cpp'],
        include_dirs=[
            get_pybind_include(),
            str(src_dir),
            eigen_include,
        ],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3'],
    ),
]


class build_ext_fixed(build_ext):
    """Custom build_ext to fix platform tags for PyPI"""
    def finalize_options(self):
        build_ext.finalize_options(self)
        # This will be set correctly when building wheels


# Read README from parent directory or current directory
readme_paths = [
    Path(__file__).parent.parent / 'README.md',  # When called from python/ directory
    Path(__file__).parent / 'README.md',          # When installed from sdist
]

long_description = "A Lightweight Incremental KD-Tree for dynamic point insertion"
for readme_path in readme_paths:
    if readme_path.exists():
        try:
            with open(readme_path, 'r', encoding='utf-8') as fh:
                long_description = fh.read()
            print(f"Using README from: {readme_path}")
            break
        except Exception as e:
            print(f"Warning: Could not read {readme_path}: {e}")

setup(
    name='likd-tree',
    version='1.0.1',
    author='Liu Yang',
    author_email='',
    description='A Lightweight Incremental KD-Tree for dynamic point insertion',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/scomup/likd-tree',
    project_urls={
        'Bug Tracker': 'https://github.com/scomup/likd-tree/issues',
        'Documentation': 'https://github.com/scomup/likd-tree#readme',
        'Source Code': 'https://github.com/scomup/likd-tree',
    },
    packages=setuptools.find_packages(),
    ext_modules=ext_modules,
    install_requires=[
        'pybind11>=2.6.0',
        'numpy>=1.19.0',
    ],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='kd-tree kdtree spatial-index nearest-neighbor point-cloud',
    include_package_data=True,
    zip_safe=False,
    cmdclass={'build_ext': build_ext_fixed},
)
