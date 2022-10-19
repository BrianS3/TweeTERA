from setuptools import setup

setup(name='TweeTERA',
      version='0.1',
      description='University of Michigan Milestone 2 Project',
      url='https://github.com/BrianS3/MI2_drown_murphy_seko',
      author='Drown, Gretchyn; Murphy, Patrick; Seko, Brian',
      author_email='bseko@umich.edu',
      license='MIT',
      packages=['TweeTERA'],
      install_requires=[
            'dotenv',
            'requests',
            'mysql-connector-python',
            'pytrends',
            'plotly',
            'gensim',
            'operator',
            'text2emotion==0.0.5',
            'emoji==1.6.3',
            'pandas',
            'numpy',
            'tqdm',
            'statsmodel',
            'datetime',
            'nbformat',
            'kaleido',
            'DateTime',
            'altair-saver',
            'collection',
            'XlsxWriter',
            'mysqlclient',
            'scikit-learn',
            'itertools',
            'nltk',
            'gensim==4.2.0',
            'langdetect==1.0.9',
            'pytrends'
      ],
      zip_safe=False)