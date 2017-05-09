from setuptools import setup

setup(name='neural_nlg_solver',
      version='1.0',
      description='Solve proportional analogies on strings using neural networks',
      author='Vivatchai KAVEETA',
      author_email='vivatchai@fuji.waseda.jp',
      packages=['neural_nlg_solver'],
      install_requires=[
          'PyQt5',
          'keras>=2.0.0',
          'tensorflow>=1.0.0',
          'matplotlib',
          'numpy',
          'scipy',
      ],
      zip_safe=False)
