from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)-> List[str]:
    '''
    This function returns the list of requirements.

    '''
    requirements = []
    file = open(file_path,"r")
    for line in file:
        if(line != "-e ."):
            requirements.append(line.replace("\n",""))
    return requirements

setup(name='recommendationSys',
      version='0.0.1',
      description='Student Performance Model',
      author='Sanket Saxena',
      author_email='sanketsaxena293@gmail.com',
      packages=find_packages(),
      install_requires = get_requirements("requirements.txt")
    )