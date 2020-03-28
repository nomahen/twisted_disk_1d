from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_modules = [Extension("evolve_fvm",
                ["evolve_fvm.pyx"],
                extra_compile_args = ["-ffast-math"])]

setup(
    name="evolve_fvm",
    cmdclass={"build_ext":build_ext},
    ext_modules=ext_modules)
