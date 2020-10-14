from setuptools import setup, find_packages
from pathlib import Path

setup(
   name='kineticsz',
   version='0.1.0',
   author='Utkarsh Giri',
   author_email='ugiri@perimeterinstitute.ca',
   packages=['kineticsz', 'kineticsz.utils'],
   #scripts = [x.as_posix() for x in list(Path('scripts').glob('*'))],
   description='Library for kSZ reconstruction and analysis',
   install_requires=[
   ],
)
