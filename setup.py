from setuptools import setup


with open("requirements.txt") as req:
    requirements = [x.strip() for x in req.readlines()]

if __name__ == "__main__":
    setup(name="mlsurfacelayer",
          version="0.1",
          description='Machine learning parameterizations for the surface layer.',
          author="David John Gagne, Tyler McCandless, and Tom Brummet",
          author_email="dgagne@ucar.edu",
          license="MIT",
          url="https://github.com/NCAR/mlsurfacelayer",
          packages=["mlsurfacelayer"],
          install_requires=requirements)
