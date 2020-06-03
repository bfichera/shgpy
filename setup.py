import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('.version', 'r') as fh:
    version = fh.read().splitlines()[0]

setuptools.setup(
    name='shgpy',
    version=version,
    author='Bryan Fichera',
    author_email='bfichera@mit.edu',
    description='A collection of utilities for analyzing SHG data.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://bfichera.github.io/shgpy/',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
