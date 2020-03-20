from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

#setup(
#        ext_modules = cythonize("evolve_kernel.pyx")
#)

ext_modules = [Extension("evolve_kernel",
                ["evolve_kernel.pyx"],
                extra_compile_args = ["-ffast-math"])]

setup(
    name="evolve_kernel",
    cmdclass={"build_ext":build_ext},
    ext_modules=ext_modules)
