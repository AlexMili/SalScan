from setuptools import find_packages, setup

setup(
    name="SalScan",
    version="0.1",
    description="Universal framework for saliency models",
    url="http://github.com/AlexMili/SalScan",
    author="AlexMili",
    platforms=["any"],
    packages=["SalScan"] + ["SalScan." + i for i in find_packages("SalScan")],
    license="MIT",
    zip_safe=True,
)
