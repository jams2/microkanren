from distutils.core import setup, Extension
import sysconfig

_DEBUG = True
_DEBUG_LEVEL = 0

extra_compile_args = sysconfig.get_config_var("CFLAGS").split()
extra_compile_args += ["-std=c11", "-Wall", "-Wextra", "-fanalyzer"]
if _DEBUG:
    extra_compile_args += ["-g3", "-O0", "-DDEBUG=%s" % _DEBUG_LEVEL, "-UNDEBUG"]
else:
    extra_compile_args += ["-DNDEBUG", "-O3"]

setup(
    ext_modules=[
        Extension(
            "mkcore",
            sources=["src/ext/mkcore_module.c"],
            extra_compile_args=extra_compile_args,
        ),
    ]
)
