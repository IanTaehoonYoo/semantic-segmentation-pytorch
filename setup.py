from setuptools import find_packages, setup


setup(name="seg_torch",
      version="0.1.0",
      description="Semantic Segmentation with Pytorch",
      author="Ian Yoo",
      author_email='thyoostar@gmail.com',
      platforms=["any"],
      license="MIT",
      url="https://github.com/IanTaehoonYoo/semantic-segmentation-pytorch",
      packages=find_packages(exclude=["segmentation/test/dataset"]),
      install_requires=[
            "torch>=1.5.0",
            "torchvision>=0.6.0",
            "opencv-python",
            "tqdm"],
      extras_require={
            "tensorflow": ["tensorflow"], #this is to provide backbone models.
      },
      classifiers=['License :: OSI Approved :: MIT License']
)