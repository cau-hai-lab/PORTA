from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
CUSPARSELT_ROOT = Path("/usr/local/lib/python3.10/dist-packages/cusparselt")

setup(
    name="cusparselt_linear_cpp",
    ext_modules=[
        CUDAExtension(
            name="cusparselt_linear_cpp",
            sources=[
                str(ROOT / "cusparselt_linear.cpp"),
                str(ROOT / "cusparselt_linear_kernel.cu"),
            ],
            include_dirs=[str(CUSPARSELT_ROOT / "include")],
            library_dirs=[str(CUSPARSELT_ROOT / "lib")],
            libraries=[":libcusparseLt.so.0"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-gencode=arch=compute_89,code=sm_89"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)

