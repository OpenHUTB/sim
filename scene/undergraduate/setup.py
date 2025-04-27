'''

Description: 
'''

from setuptools import setup, find_packages

setup(name='safebench',
      packages=["safebench"],
      include_package_data=True,
      version='1.0.0',
      install_requires=['gym', 'pygame'])
