from setuptools import setup, find_packages

setup(name='rfdiffusion',
      version='1.2.1',
      description='RFdiffusion is an open source method for protein structure generation.',
      author='Rosetta Commons',
      url='https://github.com/RosettaCommons/RFdiffusion',
      scripts=["inference/run_inference.py"],
      packages=find_packages(),
      python_requires='>=3.10,<3.13',  # Support Python 3.12 for future compatibility
      install_requires=[
          # PyTorch ecosystem - flexible versions for forward compatibility
          'torch>=2.3.0',  # No upper bound - let conda/pip resolve
          'torchvision>=0.18.0', 
          'torchaudio>=2.3.0',
          # DGL - flexible to support both 1.x stable and 2.x future
          'dgl>=1.1.0',  # No upper bound - allows source builds
          # Configuration management - relaxed bounds
          'hydra-core>=1.3.0',
          'omegaconf>=2.2.0',
          'pyrsistent>=0.20.0',
          # Scientific computing - more flexible bounds
          'numpy>=1.21.0',  # Support numpy 2.x when ecosystem ready
          'scipy>=1.9.0',
          'matplotlib>=3.5.0',
          # SE3NN and related - relaxed for compatibility
          'e3nn>=0.5.1',  # Allow newer e3nn versions
          # Monitoring and utilities - flexible bounds
          'wandb>=0.15.0',
          'pynvml>=11.5.0',
          'decorator>=5.1.1',
          # Additional utilities - no version constraints
          'tqdm',
          'requests',
      ],
      extras_require={
          'stable': [
              # Conservative versions for maximum stability
              'torch>=2.3.0,<2.5.0',
              'torchvision>=0.18.0,<0.20.0', 
              'torchaudio>=2.3.0,<2.5.0',
              'dgl>=1.1.0,<2.0.0',
              'numpy>=1.21.0,<2.0.0',
              'scipy>=1.9.0,<2.0.0',
              'e3nn>=0.5.1,<0.6.0',
              'hydra-core>=1.3.0,<2.0.0',
              'omegaconf>=2.2.0,<2.4.0',
              'pynvml>=11.5.0,<13.0.0',
              'decorator>=5.1.1,<6.0.0',
          ],
          'bleeding-edge': [
              # Latest versions with PyTorch 2.7 + CUDA 12.8 + DGL 2.4+
              'torch>=2.7.0',
              'torchvision>=0.19.0',
              'torchaudio>=2.7.0', 
              'dgl>=2.4.0',  # Build from source if needed
              'numpy>=2.0.0',  # Support numpy 2.x
              'scipy>=1.14.0',
              'e3nn>=0.6.0',
          ],
          'dgl-source': [
              # Dependencies for building DGL from source
              'cmake>=3.18',
              'pybind11>=2.6.0',
              'torch>=2.7.0',  # PyTorch 2.7 for DGL 2.4+ compatibility
              'numpy>=2.0.0',
              'scipy>=1.14.0',
              'networkx>=2.1',
              'requests>=2.19.0',
              'tqdm',
              'psutil>=5.8.0',
          ],
          'cuda129': [
              # CUDA 12.9 optimized versions
              'torch==2.3.0',
              'torchvision==0.18.0',
              'torchaudio==2.3.0',
              'dgl==1.1.3',
              'numpy>=1.21.0,<2.0.0',
          ],
          'cuda128-dgl24': [
              # PyTorch 2.7 + CUDA 12.8 + DGL 2.4+ combination
              # Note: Install PyTorch 2.7 CUDA 12.8 manually with:
              # pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
              'torch>=2.7.0',
              'torchvision>=0.22.0',
              'torchaudio>=2.7.0',
              'dgl>=2.4.0',  # Build from source
              'numpy>=2.0.0',
              'scipy>=1.14.0',
          ],
          'cuda121': [
              # CUDA 12.1 fallback
              'torch==2.3.0',
              'torchvision==0.18.0',
              'torchaudio==2.3.0',
              'dgl==1.1.3',
          ],
          'cuda111': [
              # Legacy CUDA 11.1 support
              'torch>=1.9.0,<2.0.0',
              'torchvision',
              'torchaudio',
              'dgl>=1.0.0,<2.0.0',
          ],
          'dev': [
              # Development dependencies
              'pytest>=7.0',
              'black>=22.0',
              'flake8>=5.0',
              'mypy>=1.0',
              'pre-commit>=2.20',
              'jupyter>=1.0',
          ],
      },
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Chemistry',
      ],
)