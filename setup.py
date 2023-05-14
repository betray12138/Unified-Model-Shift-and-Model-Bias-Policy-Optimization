from distutils.core import setup
from setuptools import find_packages

setup(
    name='mbpo',
    packages=find_packages(),
    version='0.0.1',
    description='Unified Model Shift and Model Bias Policy Optimization',
    author='Hai Zhang',
    author_email='zhanghai12138@tongji.edu.cn',
    entry_points={
        'console_scripts': (
            'mbpo=softlearning.scripts.console_scripts:main',
            'viskit=mbpo.scripts.console_scripts:main'
        )
    },
    requires=(),
    zip_safe=True,
    license='MIT'
)