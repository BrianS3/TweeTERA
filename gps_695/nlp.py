def clean_tweets(df):
    '''
    INPUT: Pandas DataFrame
    Removes tweets that are sent when a person posts a video or photo only;
    removes URLS, username mentions from tweet text;
    translates non-English Tweets;
    isolates hashtags;
    OUTPUT: original df with added columns TIDY_TWEET, HASHTAGS
    '''
    import re
    from langdetect import detect
    import numpy as np

    data = df.copy()

    data = data[data.TWEET_TEXT.str.contains("Just posted a")==False] # posts that are only photos/videos

    hashtags = []
    tweets = []
    for tweet in data['TWEET_TEXT']:
        tweet=str(tweet)
        tweet = tweet.lower().strip()
        hashtags.append(re.findall(r"\B#\w*[a-zA-Z]+\w*", tweet.lower().strip())) # isolate hashtags
        tweet = re.sub("[0-9]", "", tweet) # digits
        tweet = re.sub("@[\w]*", "", tweet) # usernames
        tweet = re.sub(r"https?:\/\/.*[\r\n]*", "", tweet) # URLs
        tweet = re.sub(r"[^\w'\s]+", " ", tweet) # punctuation
        tweet = tweet.replace("rt", "")

        # Detect language
        if tweet.strip() == "":
            # making empty tweets non-English so they are removed; detect won't process empty strings
            tweet = "donde está el baño"
        tweet = tweet.lstrip()
        tweet = tweet.rstrip()
        lang = detect(tweet)

        if lang != 'en':
            tweet="!!!DELTEME!!!"

        tweets.append(tweet)

    data['TIDY_TWEET'] = tweets
    data['HASHTAGS'] = hashtags
    data=data[data['TIDY_TWEET']!='!!!DELTEME!!!']

    return data

def lemmatize(df):
    '''INPUT: df with tidy_tweet column
    tokenizes;
    removes stopwords and lingering URLS;
    lemmatizes
    RETURNS: original df with added LEMM column
    '''
    from nltk.corpus import stopwords

    # tokenize
    tokens = []
    for i, item in enumerate(df['TIDY_TWEET']):
        tokens.append(item.split())

    # remove URLs
    for item in tokens:
        for thing in item:
            if 'http' in thing:
                item.remove(thing)

    # remove stopwords
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words('english'))
    new_tokens = []
    for item in tokens:
        new = [i for i in item if not i in stop_words]
        new_tokens.append(new)

    # lemmatize
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    lemms = []
    for item in new_tokens:
        l = []
        for word in item:
            lemm = lemmatizer.lemmatize(word.lower())
            l.append(lemm)
        if l != 'rt':
            lemms.append(l)
            
    for lemm_list in lemms:
        lemm_list = [item for item in lemm_list if item != 'rt']

    df['LEMM'] = lemms


def analyze_tweets():
    '''
    Analyzes tweets for sentiment analysis
    Get emotions: Happy, Angry, Surprise, Sad, Fear, *Neutral, *Mixed
    :return: None, database loaded with data
    '''
    import text2emotion as te
    from gps_695 import database as d
    import pandas as pd
    from tqdm import tqdm

    cnx = d.connect_to_database()
    get_tweets_query = 'SELECT TWEET_ID, TIDY_TWEET FROM TWEET_TEXT'
    df_full = pd.read_sql_query(get_tweets_query, cnx)
    df = df_full.sample(round(len(df_full)*.2))

    # get emotion scores and predominant tweet emotion(s)
    emos = []
    for item in df['TIDY_TWEET']:
        emo = te.get_emotion(item)
        emos.append(emo)
    df['TWEET_EMO'] = emos

    predominant_emotion = []
    pred_emo_score = [] ##
    for item in tqdm(df['TWEET_EMO']):
        sort_by_score_lambda = lambda score: score[1]
        sorted_value_key_pairs = sorted(item.items(), key=sort_by_score_lambda, reverse=True)

        emos = []
        emo_scores = [] ##
        if sorted_value_key_pairs[0][1] == 0:
            emos.append('Neutral')
            emo_scores.append(0) ##
        else:
            emos.append(sorted_value_key_pairs[0][0])
            emo_scores.append(sorted_value_key_pairs[0][1]) ##
        for i, item in enumerate(sorted_value_key_pairs):
            a = sorted_value_key_pairs[0][1]
            if sorted_value_key_pairs[i][1]==a and i!=0 and a!=0:
                emos.append(sorted_value_key_pairs[i][0])

        predominant_emotion.append(emos)
        pred_emo_score.append(emo_scores) ##

    for i, item in enumerate(predominant_emotion):
        if len(item)>1:
            predominant_emotion[i] = ['Mixed']

    predominant_emotion = [element for sublist in predominant_emotion for element in sublist]
    pred_emo_score = [element for sublist in pred_emo_score for element in sublist] ##
    df['OVERALL_EMO'] = predominant_emotion
    df['OVERALL_EMO_SCORE'] = pred_emo_score
    df = df[['TWEET_ID', 'OVERALL_EMO']]

    column_list = list(df.columns)

    cnx = d.connect_to_database()
    for ind, row in tqdm(df.iterrows()):
        query = (f"""
                    UPDATE TWEET_TEXT
                    SET OVERALL_EMO = '{row[column_list[1]]}'
                    WHERE TWEET_ID = '{row[column_list[0]]}';
                    """)
        cnx.execute(query)
    cnx.close()

def get_associated_keywords(df, search_term, perc_in_words=0.1, **kwargs):
    '''
    Function finds the associated keywords from the initial data load
    :param df: df with LEMM column
    :param search_term: the search term associated with the news event/tweets
    :param returned_items: integer value to specify how many keywords max you want returned
    :param perc_in_words: the smallest threshold required for word frequency. If this is set to 10%, then 10% of all words must have the terms.
    :param **kwargs: keyword arguments from sklearn NMF model for grid search
    Lowering this value produces more variety.
    :return: list of strings representing keywords associated with search_term
    '''
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    import pandas as pd

    df2 = df.copy()
    for ind, row in df2.iterrows():
        lemm2 = "".join(row['LEMM'].replace("[", "").replace("'", "").replace("]", '').replace(",", ""))
        row['LEMM'] = lemm2

    # Perform NMF unsupervised learning to find topics
    vect = TfidfVectorizer(min_df=int(np.round(perc_in_words * len(df))), stop_words='english', ngram_range=(2, 2))

    # term must appear in 10% of tweets, looking for bigrams
    result = {}

    try:
        X = vect.fit_transform(df2.LEMM)
        model = NMF(**kwargs, max_iter=1000)
        model.fit(X)
        components_df = pd.DataFrame(model.components_, columns=vect.get_feature_names_out())
        for topic in range(components_df.shape[0]):
            tmp = components_df.iloc[topic]
            #associated_keywords = list(tmp.nlargest().index)
            coherency_scores = dict(tmp.nlargest())
            temp_dict = {k: v for k, v in coherency_scores.items() if search_term.lower() not in k}
            result.update(temp_dict)
        return result

    except ValueError:
        return "Could not find associated topics."

def gridsearch(search_term):
    """
    Function performs a grid search to optimize the model parameters for obtaining associated keywords.
    :param search_term: search term used in load_tweets function
    :return: top two keywords as a list with lowest MSE from google trends, to search term.
    """
    import pandas as pd
    from gps_695 import database as d
    from gps_695 import nlp as n
    from itertools import product
    from collections import Counter

    cnx = d.connect_to_database()
    query = f"""select LEMM from TWEET_TEXT where lower(SEARCH_TERM) = '{search_term}';"""
    df = pd.read_sql_query(query, cnx)

    alphas = [0, 0.5, 1]
    l1_ratios = [0, 5, 10]
    percents = [0.1, 0.05, 0.01]

    grid_search_params = list(product(alphas, l1_ratios, percents))

    param_df = pd.DataFrame(
        grid_search_params,
        columns=['alpha', 'l1_ratio', 'percents'])

    file = open('output_data/word_association_eval.txt', 'w', encoding="utf-8")
    grid_search_results = Counter()

    for ind, row in param_df.iterrows():
        alpha_val = row['alpha']
        l1_val = row['l1_ratio']
        perc_val = row['percents']

        file = open('output_data/word_association_eval.txt', 'a')
        file.write(f"{ind} -- alpha:{alpha_val}  l1_ratio:{l1_val} perc in words:{perc_val} \n")

        kw_list = n.get_associated_keywords(df, search_term, perc_in_words=perc_val.astype(float),
                                            alpha_W=alpha_val.astype(float), l1_ratio=l1_val.astype(int))
        if kw_list != 'Could not find associated topics.':
            for k,v in kw_list.items():
                file.write(f"{k.encode('utf8')}")
                file.write("\n")
            file.close()

        grid_search_results.update(kw_list)

    associated_words_df = pd.DataFrame(grid_search_results.most_common(), columns=['term', 'score'])
    associated_words_df = associated_words_df[associated_words_df['term'].str.len()>1]
    associated_words_df.to_csv('output_data/word_association_results.csv')
    associated_words = list(associated_words_df.sort_values('score', ascending=False)[:2]['term'])

    return associated_words

def create_sentiment_model():
    """
    Function creates a supervised sentiment prediction model. This is used to decrease load times by predicting tweet sentiment vs a traditional sentiment analysis. Function pushes sentinments to current database.
    :return: None
    """

    file = open('output_data/sentiment_optimization.txt', 'w+')

    import pandas as pd
    from gps_695 import credentials as c
    from gps_695 import database as d
    from sklearn.metrics import accuracy_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GridSearchCV, KFold
    from tqdm import tqdm

    c.load_env_credentials()
    cnx = d.connect_to_database()

    query = """
    SELECT
    LEMM
    ,OVERALL_EMO
    FROM TWEET_TEXT
    WHERE OVERALL_EMO IS NOT NULL
    """
    df = pd.read_sql_query(query, cnx)

    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, use_idf=True)
    X = vectorizer.fit_transform(df['LEMM'])
    y = df['OVERALL_EMO']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    svm = SVC(kernel='rbf', gamma=5)

    kernel = ['linear', 'sigmoid', 'rbf']
    C = [0, 1, 2, 3, 5 , 10]
    gamma = [0.01, 0.05, .1, 0.5, 1, 5, 10]

    param_grid = dict(kernel=kernel,C=C,gamma=gamma)

    grid = GridSearchCV(estimator=svm, param_grid=param_grid,cv=KFold(), verbose=10)

    grid_results = grid.fit(X_train, y_train)
    file.write("Best: {0}, using {1} \n".format(grid_results.best_score_, grid_results.best_params_))

    gpred = grid.predict(X_test)
    file.write(f"f1 score: {f1_score(y_test, gpred, average='micro')} \n")
    file.write(f"accuracy: {accuracy_score(y_test, gpred)} \n")
    file.close()

    cnx = d.connect_to_database()
    query2 = """
    SELECT TWEET_ID, LEMM
    FROM TWEET_TEXT
    WHERE OVERALL_EMO IS NULL
    """
    df2 = pd.read_sql_query(query2, cnx)
    X2 = vectorizer.transform(df2['LEMM'])

    gpred2 = grid.predict(X2)

    df2['pred'] = gpred2.tolist()
    column_list = list(df2.columns)

    cnx = d.connect_to_database()
    for ind, row in tqdm(df2.iterrows()):
        query = (f"""
                    UPDATE TWEET_TEXT
                    SET OVERALL_EMO = '{row[column_list[2]]}'
                    WHERE TWEET_ID = '{row[column_list[0]]}';
                    """)
        cnx.execute(query)
    cnx.close()
