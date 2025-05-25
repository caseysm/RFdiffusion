from setuptools import setup, find_packages

setup(name='rfdiffusion',
      version='1.2.1',
      description='RFdiffusion is an open source method for protein structure generation.',
      author='Rosetta Commons',
      url='https://github.com/RosettaCommons/RFdiffusion',
      scripts=["scripts/run_inference.py"],
      packages=find_packages(),
      python_requires='>=3.10,<3.12',
      install_requires=[
          # PyTorch ecosystem - stable versions compatible with DGL
          'torch>=2.3.0,<2.4.0',
          'torchvision>=0.18.0,<0.19.0', 
          'torchaudio>=2.3.0,<2.4.0',
          # DGL - using 1.x for RFdiffusion compatibility (avoids torchdata issues)
          'dgl>=1.1.0,<2.0.0',
          # Configuration management
          'hydra-core>=1.3.0,<2.0.0',
          'omegaconf>=2.2.0,<2.4.0',
          'pyrsistent>=0.20.0',
          # Scientific computing with upper bounds for stability
          'numpy>=1.21.0,<2.0.0',
          'scipy>=1.9.0,<2.0.0',
          'matplotlib>=3.5.0',
          # SE3NN and related
          'e3nn>=0.5.1,<0.6.0',
          # Monitoring and utilities
          'wandb>=0.15.0',
          'pynvml>=11.5.0,<13.0.0',
          'decorator>=5.1.1,<6.0.0',
          # Additional utilities
          'tqdm',
          'requests',
      ],
      extras_require={
          'cuda129': [
              # Specific versions for CUDA 12.9 setup
              'torch==2.3.0',
              'torchvision==0.18.0',
              'torchaudio==2.3.0',
              'dgl==1.1.3',
          ],
          'cuda121': [
              # Fallback for CUDA 12.1 if needed
              'torch==2.3.0',
              'torchvision==0.18.0',
              'torchaudio==2.3.0',
              'dgl==1.1.3',
          ],
          'cuda111': [
              # Legacy CUDA 11.1 support
              'torch==1.9.*',
              'torchvision',
              'torchaudio',
              'dgl<2.0.0',
          ],
          'dev': [
              # Development dependencies
              'pytest>=6.0',
              'black',
              'flake8',
              'mypy',
          ],
      },
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Chemistry',
      ],
)