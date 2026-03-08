from setuptools import setup, find_packages

setup(
    name="anomaly_project",
    version="0.1",
    packages=find_packages(where="src"),  # Look for packages in src
    package_dir={"": "src"},             # Root of packages is src
)