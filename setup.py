# Always prefer setuptools over distutils
import glob
import os
import pathlib
import shutil

from setuptools import setup, Command,find_packages
from os.path import join, dirname

# https://github.com/pypa/setuptools/issues/1347
class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    CLEAN_FILES = './build ./dist ./*.pyc ./*.tgz ./*.egg-info'.split(' ')

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        global here

        for path_spec in self.CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(here, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(str(here)):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, here))
                # print('removing %s' % os.path.relpath(path))
                shutil.rmtree(path)



setup(
    name='django-ovation-prime',
    version='0.1.1',
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
    cmdclass={
        'clean': CleanCommand,
        # 'install': CustomInstallCommand,
    },

)
