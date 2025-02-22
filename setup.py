#
# Setup file for
# algo -- Kareem's Algorithmic Trading
#
# (c) Kareem Powell
#
from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='algo',
      version='0.0.56',
      description='algo Algorithmic Trading with Kareem',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Kareem Powell',
      author_email='#',
      url='#',
      packages=['algo'],
      install_requires=[
      ]
      )
