from setuptools import setup, find_packages
if __name__ == '__main__':
    setup(name="TrueMultimer",
          version="0.1",
          packages=find_packages(include=["alphafold*"]),
          )
