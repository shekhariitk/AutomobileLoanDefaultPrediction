from setuptools import setup, find_packages
from typing import List


HYPHEN_E_DOT = "-e."


NAME = "AutomobileLoanDefaultPrediction"
VERSION = "0.0.1"
AUTHOR = "Shekhar"
AUTHOR_EMAIL = "s.sumanpathak513@gmail.com"
DESCRIPTION = "A package to predict the default of automobile loan"


def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [ req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

        return requirements

        



setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    install_requires  = get_requirements("requirements.txt"),
    packages=find_packages()
)