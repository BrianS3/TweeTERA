def install_dependencies():
    import sys
    import subprocess

    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','numpy'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','pandas'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','langdetect==1.0.9'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','text2emotion==0.0.5'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','gensim==4.2.0'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','nltk'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','emoji==1.6.3'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','pytest-shutil'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','requests'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','scikit-learn'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','mysql-client'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','SQLAlchemy'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','mysql-connector-python'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','mysqlclient'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','pytrends'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','python-dotenv'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','plotly'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','operator'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','tqdm'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','nbformat'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','statsmodel'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','kaleido'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','DateTime'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','altair-saver'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','collection'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','itertools'])

    import nltk
    nltk.download('omw-1.4')