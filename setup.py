from setuptools import setup, find_packages
from os.path import join, dirname

setup(
    name='django-ovation-prime',
    version='0.1.0',
    install_requires=[
        'Django >= 3.2.0',
        'numpy >= 1.21.0',
        'OvationPyme @ git+https://github.com/mishka251/OvationPyme.git@0.1.2#egg=ovationpyme',
        'nasaomnireader @ git+https://github.com/mishka251/nasaomnireader.git@0.1.2#egg=nasaomnireader'
    ],
    dependency_links=[
        # 'git+https://github.com/mishka251/OvationPyme.git@0.1.1#egg=ovationpyme',
        # 'git+https://github.com/mishka251/nasaomnireader.git@0.1.1#egg=nasaomnireader'
    ],
    packages=find_packages(),
    long_description=open(join(dirname(__file__), 'readme.md')).read(),
)
