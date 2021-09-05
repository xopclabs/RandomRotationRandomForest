from pathlib import Path
from setuptools import setup

# The directory containing this file
HERE = Path(__file__).parent

# The text of the README file
README = (HERE / 'README.md').read_text()

# This call to setup() does all the work
setup(
    name='rrsklearn',
    version='1.0.0',
    description='Random Rotation implementation for sklearn\'s Random Forests',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/xopclabs/random-rotation-sklearn',
    author='Pavel Leyba',
    author_email='leybapavel@gmail.com',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    packages=['rrsklearn'],
    include_package_data=True,
    install_requires=['numpy', 'sklearn']
)
