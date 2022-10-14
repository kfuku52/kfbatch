from setuptools import setup, find_packages
import os
import re
import ast

with open(os.path.join('kfbatch', '__init__.py')) as f:
        match = re.search(r'__version__\s+=\s+(.*)', f.read())
version = str(ast.literal_eval(match.group(1)))

setup(
        name             = 'kfbatch',
        version          = version,
        description      = 'Batch job management',
        license          = "BSD 3-clause License",
        author           = "Kenji Fukushima",
        author_email     = 'kfuku52@gmail.com',
        url              = 'https://github.com/kfuku52/kfbatch.git',
        keywords         = 'phylogenetics',
        packages         = find_packages(),
        install_requires = [],
        scripts          = ['kfbatch/kfbatch',],
        include_package_data = False,
)