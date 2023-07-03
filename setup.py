<<<<<<< HEAD
from setuptools import setup
if __name__ == '__main__':
    setup()
=======
from setuptools import setup, find_packages
if __name__ == '__main__':
    setup(name="TrueMultimer",
          version="0.1",
          packages=find_packages(include=["alphafold*"]),
          )
>>>>>>> 4425307c0570304cfdf3411b13525817ebebb01b
