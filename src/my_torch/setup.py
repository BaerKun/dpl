from setuptools import setup, find_packages

setup(
    name="my_torch",            # 包名称
    version="1.0",               # 版本号
    packages=find_packages(),    # 自动查找包中的模块
    install_requires=["numpy"],         # 列出包的依赖（如有）
    author="Bear Kun",
    description="A deep learning library implemented with NumPy that mimics PyTorch",
)
