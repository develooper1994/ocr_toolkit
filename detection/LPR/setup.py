import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='platerecognition',  
     version='0.0.6',
     description="An LPR package",
     long_description=long_description,
     long_description_content_type="text/markdown",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: TurkAI License",
         "Operating System :: OS Independent",
     ],
     install_requires=required,
 )
