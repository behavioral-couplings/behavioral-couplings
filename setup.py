from setuptools import find_packages, setup


setup(
    name="behavioral_coupling",
    version="0.1.0",
    description="behavioral_coupling research utilities",
    packages=find_packages(),
    package_data={"behavioral_coupling.utils": ["ds_zero*.json"]},
    install_requires=[],
    python_requires=">=3.12",
)
