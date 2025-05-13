# -*- coding: utf-8 -*-

import sys
import subprocess
import re

from setuptools import setup, find_packages
from setuptools.command.sdist import sdist as _sdist
from setuptools.command.install import install as _install
import hicgan._version as version

# Function to parse requirements.txt
def parse_requirements(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        return [line.strip() for line in lines if line and not line.startswith('#')]

# Reading dependencies
requirements = parse_requirements('requirements.txt')


def get_version():
    try:
        f = open("hicgan/_version.py")
    except EnvironmentError:
        return None
    for line in f.readlines():
        mo = re.match("__version__ = '([^']+)'", line)
        if mo:
            ver = mo.group(1)
            return ver
    return None


class sdist(_sdist):

    def run(self):
        # update_version_py()
        self.distribution.metadata.version = get_version()
        return _sdist.run(self)

# Install class to check for external dependencies from OS environment


class install(_install):

    def run(self):
        # update_version_py()
        self.distribution.metadata.version = get_version()
        _install.run(self)
        return

    def checkProgramIsInstalled(self, program, args, where_to_download,
                                affected_tools):
        try:
            subprocess.Popen([program, args],
                             stderr=subprocess.PIPE,
                             stdout=subprocess.PIPE)
            return True
        except EnvironmentError:
            # handle file not found error.
            # the config file is installed in:
            msg = "\n**{0} not found. This " \
                  "program is needed for the following "\
                  "tools to work properly:\n"\
                  " {1}\n"\
                  "{0} can be downloaded from here:\n " \
                  " {2}\n".format(program, affected_tools,
                                  where_to_download)
            sys.stderr.write(msg)

        except Exception as e:
            sys.stderr.write("Error: {}".format(e))

setup(
    name='Hi-cGAN',
    version=get_version(),
    author='Ralf Krauth, Anup Kumar, Joachim Wolff',
    author_email='wolff.joachim@mh-hannover.de',
    long_description_content_type='text/markdown',
    url='https://github.com/joachimwolff/hi-cgan',
    packages=find_packages(),
    install_requires=requirements,
    scripts=['bin/hicTraining', 'bin/hicPredict', 'bin/hicComputeCorrelation', 'bin/hicOptimizer', 'bin/hicScoring', 'bin/hicFeatureSelection'],
    include_package_data=True,
    package_dir={'hicgan': 'hicgan'},
    license='LICENSE',
    description='Set of programs to process, analyze and visualize Hi-C data',
    long_description=open('README.md').read(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics'],
    zip_safe=False,
    cmdclass={'sdist': sdist, 'install': install}
)