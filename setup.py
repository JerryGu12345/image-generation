from setuptools import find_packages,setup

def get_requirements():
    requirements=[]
    with open("requirements.txt") as file_obj:
        for req in file_obj.readlines():
            req=req.replace("\n","")
            if req[0]!='-':
                requirements.append(req)
    print(requirements)
    return requirements

setup(
    name="Image_Generation",
    version='0.0.1',
    author="Jerry Gu",
    author_email="jerrygu@live.com",
    packages=find_packages(),
    install_requires=get_requirements()
)