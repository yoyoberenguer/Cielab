# encoding: utf-8

"""
Setup.py file

Configure the project, build the package and upload the package to PYPI


python_version setup.py sdist bdist_wheel (to include the source)

[TEST PYPI]
repository = https://test.pypi.org/

[PRODUCTION]
repository = https://upload.pypi.org/legacy/
"""
# twine upload --repository testpypi dist/*

import setuptools

try:
    import Cython
except ImportError:
    raise ImportError("\n<Cython> library is missing on your system."
                      "\nTry: \n   C:\\pip install Cython")

try:
    import numpy
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
                      "\nTry: \n   C:\\pip install numpy")

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

try:
    import pygame
except ImportError:
    raise ImportError("\n<pygame> library is missing on your system."
                      "\nTry: \n   C:\\pip install pygame")

from Cython.Build import cythonize
from setuptools import Extension
import platform
import warnings
import sys
from config import THREAD_NUMBER, OPENMP, OPENMP_PROC, LANGUAGE, __VERSION__, TEST_VERSION, \
    __TVERSION__

# print("\n---PYTHON COPYRIGHT---\n")
# print(sys.copyright)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# NUMPY IS REQUIRED
try:
    import numpy
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
                      "\nTry: \n   C:\\pip install numpy on a window command prompt.")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

ext_link_args = ""
PythonVerMax = 99
py_requires = "Cielab requires python3 version 3.6 - %s" % ("3." + str(PythonVerMax - 1))
# If you are building the project from source with a python version
# > python 3.11 you can extend the range to force the build process
# e.g py_minor_versions = [x for x in range(6, 15)] ** compatible to python 3.14
py_minor_versions = [x for x in range(6, PythonVerMax)]

if hasattr(sys, 'version_info'):
    try:
        if not hasattr(sys.version_info, "major") \
                or not hasattr(sys.version_info, "minor"):
            raise AttributeError
        py_major_ver = sys.version_info.major
        py_minor_ver = sys.version_info.minor
    except AttributeError:
        raise SystemExit(py_requires)
else:
    raise SystemExit(py_requires)

if py_major_ver != 3 or py_minor_ver not in py_minor_versions:
    raise SystemExit(
        "Cielab support python3 versions 3.6 - %s got version %s"
        % (("3." + str(PythonVerMax - 1)), str(py_major_ver) + "." + str(py_minor_ver)))

if hasattr(platform, "architecture"):
    arch = platform.architecture()
    if isinstance(arch, tuple):
        proc_arch_bits = arch[0].upper()
        proc_arch_type = arch[1].upper()
    else:
        raise AttributeError("Platform library is not install correctly")
else:
    raise AttributeError("Platform library is missing attribute <architecture>")

if hasattr(platform, "machine"):
    machine_type = platform.machine().upper()
else:
    raise AttributeError("Platform library is missing attribute <machine>")

if hasattr(platform, "platform"):
    plat = platform.platform().upper()

else:
    raise AttributeError("Platform library is missing attribute <platform>")

if plat.startswith("WINDOWS"):
    ext_compile_args = ["/openmp" if OPENMP else "", "/Qpar", "/fp:fast", "/O2", "/Oy", "/Ot"]


elif plat.startswith("LINUX"):
    if OPENMP:
        ext_compile_args = \
            ["-DPLATFORM=linux", "-march=i686" if proc_arch_bits == "32BIT" else "-march=x86-64",
             "-m32" if proc_arch_bits == "32BIT" else "-m64", "-O3", "-ffast-math", "--param=max-vartrack-size=1500000",
             "-Wall", OPENMP_PROC, "-static"]
        ext_link_args = [OPENMP_PROC]
    else:
        ext_compile_args = \
            ["-DPLATFORM=linux", "-march=i686" if proc_arch_bits == "32BIT" else "-march=x86-64",
             "-m32" if proc_arch_bits == "32BIT" else "-m64", "-O3", "-ffast-math", "-Wall", "-static",
             "--param=max-vartrack-size=1500000"]
        ext_link_args = ""

# todo add support of openmp for MAC
elif plat.startswith("MAC") or plat.startswith("DARWIN"):
    if OPENMP:
        ext_compile_args = \
            ["-march=i686" if proc_arch_bits == "32BIT" else "-march=x86-64",
             "-m32" if proc_arch_bits == "32BIT" else "-m64", "-O3", "-Wall"]
        ext_link_args = ""
    else:
        ext_compile_args = \
            ["-march=i686" if proc_arch_bits == "32BIT" else "-march=x86-64",
             "-m32" if proc_arch_bits == "32BIT" else "-m64", "-O3", "-Wall"]
        ext_link_args = ""
else:
    raise ValueError("Cielab can be build on Windows and Linux systems only.")

print("\n---COMPILATION---\n")
print("SYSTEM                : %s " % plat)
print("BUILD                 : %s " % proc_arch_bits)
print("FLAGS                 : %s " % ext_compile_args)
print("EXTRA LINK FLAGS      : %s " % ext_link_args)
print("LANGUAGE              : %s " % LANGUAGE)
print("MULTITPROCESSING      : %s " % OPENMP)
print("MULTITPROCESSING FLAG : %s " % OPENMP_PROC)
if OPENMP:
    print("MAX THREADS           : %s " % THREAD_NUMBER)

print("\n")
print("PYTHON VERSION        : %s.%s " % (sys.version_info.major, sys.version_info.minor))
print("SETUPTOOLS VERSION    : %s " % setuptools.__version__)
print("CYTHON VERSION        : %s " % Cython.__version__)
print("NUMPY VERSION         : %s " % numpy.__version__)
print("PYGAME VERSION        : %s " % pygame.__version__)

try:
    print("SDL VERSION           : %s.%s.%s " % pygame.version.SDL)
except:
    pass  # ignore SDL versioning issue

if TEST_VERSION:
    print("\n*** BUILDING Cielab TEST VERSION ***  : %s \n" % __TVERSION__)
else:
    print("\n*** BUILDING Cielab VERSION ***  : %s \n" % __VERSION__)

# define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
setuptools.setup(
    name="Cielab",
    version=__VERSION__,  # testing version "1.0.27",
    author="Yoann Berenguer",
    author_email="yoyoberenguer@hotmail.com",
    description="Cielab color space",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yoyoberenguer/Cielab",
    # packages=setuptools.find_packages(),
    packages=['Cielab'],
    ext_modules=cythonize(module_list=[

        Extension("Cielab.Cielab", ["Cielab/Cielab.pyx"],
                  extra_compile_args=ext_compile_args, extra_link_args=ext_link_args,
                  language=LANGUAGE)
    ]),

    include_dirs=[numpy.get_include()],
    license='GNU General Public License v3.0',

    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Cython",

        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries'
    ],

    install_requires=[
        'setuptools>=49.2.1',
        'Cython>=0.28',
        'numpy>=1.18',
        'pygame>=2.0'


    ],
    python_requires='>=3.6',
    platforms=['Windows'],
    include_package_data=True,
    data_files=[
        ('./lib/site-packages/Cielab',
         ['LICENSE',
          'MANIFEST.in',
          'pyproject.toml',
          'README.md',
          'requirements.txt',
          'Cielab/__init__.py',
          'Cielab/__init__.pxd',
          'Cielab/Cielab.pxd',
          'Cielab/Cielab.pyx',
          'Cielab/config.py'
          ]),
        ('./lib/site-packages/Cielab/Include',
         ['Cielab/Include/Cielab_c.c'
          ]),
        ('./lib/site-packages/Cielab/tests',
         [
             'Cielab/tests/test_cielab.py',
             'Cielab/tests/profiler.py',
             'Cielab/tests/__init__.py'
         ]),
        ('./lib/site-packages/Cielab/Assets',
         [
            'Cielab/Assets/adobe_surface.png',
            'Cielab/Assets/background2.png',
            'Cielab/Assets/background32.png',
            'Cielab/Assets/cielab_surface.png',
            'Cielab/Assets/demo.png',
            'Cielab/Assets/white_surf.png'
         ]),
        ('./lib/site-packages/Cielab/Demo',
         [
             'Cielab/Demo/__init__.py',
             'Cielab/Demo/demo.py'
         ])
    ],

    project_urls={  # Optional
        'Bug Reports': 'https://github.com/yoyoberenguer/Cielab/issues',
        'Source': 'https://github.com/yoyoberenguer/Cielab',
    }
)
