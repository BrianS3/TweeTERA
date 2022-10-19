from setuptools import setup

setup(name='TweeTERA',
      version='1.0.3',
      description='Twitter Sentiment and Analysis Package',
      include_package_data = True,
      long_description="""
      TweetERA (Tweet Emotional Response Analysis) was designed to simplify how Twitter data is analyzed. 
      This package will create a MySQL database and load Twitter data to it. 
      It will also perform a sentiment analysis on the tweets, encouraging users to run analyze new data frequently. 
      Simply enter your keyword or phrase and let the package do the rest.
      """,
      long_description_content_type='text/markdown',
      url='https://github.com/BrianS3/TweeTERA',
      download_url='https://github.com/BrianS3/TweeTERA/archive/refs/tags/v1.0.3.tar.gz',
      author='Drown, Gretchyn; Murphy, Patrick; Seko, Brian',
      author_email='bseko@umich.edu',
      license='MIT',
      py_modules = ['credentials', 'database', 'nlp', 'visuals'],
      install_requires=[
            'python-dotenv',
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