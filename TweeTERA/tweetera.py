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


def create_env_variables(db_user=None, db_pass=None, api_bearer=None, db_host=None, database=None):
    """
    Creates .env variables for database and API credentials.
    This saves an .env file to your current working directory.
    Leaving a variable as "None" will skip loading into env file. Use this to load
    variables after creating an initial .env file.
    :param db_user: username for mysql database
    :param db_pass: password for mysql database
    :param api_bearer: API bearer token
    :param db_host: server information for database
    :param database: name of mysql database, example "databse1" or "master"
    :return: None
    """
    def write_action(match):
        """
        removes credential line so you can re-write it
        :param match:credential match
        :return: None
        """
        with open(".env", "r") as fr:
            lines = fr.readlines()
        with open(".env", "w") as fw:
            for line in lines:
                if not line.startswith(f'{match}'):
                    fw.write(line)

    if db_user != None:
        write_action('mysql_username')
        with open(".env", "a+") as f:
            f.write(f"mysql_username={db_user}")
            f.write("\n")
    if db_pass != None:
        write_action('mysql_pass')
        with open(".env", "a+") as f:
            f.write(f"mysql_pass={db_pass}")
            f.write("\n")
    if api_bearer != None:
        write_action('twitter_bearer')
        with open(".env", "a+") as f:
            f.write(f"twitter_bearer={api_bearer}")
            f.write("\n")
    if db_host != None:
        write_action('db_host')
        with open(".env", "a+") as f:
            f.write(f"db_host={db_host}")
            f.write("\n")
    if database != None:
        write_action('database')
        with open(".env", "a+") as f:
            f.write(f"database={database}")
            f.write("\n")

def load_env_credentials():
    """
    Sets variables in .env file for current session.
    :return: None
    """
    import os
    with open(".env", "r") as f:
        for line in f.readlines():
            try:
                key, value = line.split('=')
                os.environ[key] = value.strip()
            except ValueError:
                # syntax error
                pass

def connect_to_database():
    """
    Creates connection to mysql database.
    :return: connection objects
    """
    import os
    import sqlalchemy
    connection_string = f'mysql://{os.getenv("mysql_username")}:{os.getenv("mysql_pass")}@{os.getenv("db_host")}:3306/{os.getenv("database")}'
    engine = sqlalchemy.create_engine(connection_string)
    try:
        cnx = engine.connect()
    except:
        print('Credentials not loaded, use credentials.load_env_credentials()')

    return cnx

def create_mysql_database():
    """
    Function creates mysql database to store twitter data.
    :return: None
    """
    file = open('TweeTERA/database_table_creation.sql', 'r')
    sql = file.read()
    file.close

    cnx = connect_to_database()
    cnx.execute(sql)
    cnx.close()

def reset_mysql_database():
    """
    Function resets mysql database for new data loading. The process will remove all tables from the database and recreate it.
    :return: None
    """

    file = open('TweeTERA/database_clear.sql', 'r')
    sql = file.read()
    file.close

    cnx = connect_to_database()
    cnx.execute(sql)
    cnx.close()
    create_mysql_database()
    print("Database fully reset")

def call_tweets(keyword, start_date, end_date, results):
    """
    Pulls tweets from research project API v2
    :param keyword: keyword of tweet for API query
    :param start_date: start date of query, YYYY-MM-DD format, string
    :param end_date: end date of query,  YYYY-MM-DD format, string
    :param results: number of results to return, max 500, int
    :return: json object
    """
    import requests
    import os
    print("Calling tweets...")
    search_api_bearer = os.getenv('twitter_bearer')
    url = f"https://api.twitter.com/2/tweets/search/all?query={keyword}&start_time={start_date}T00:00:00.000Z&end_time={end_date}T00:00:00.000Z&max_results={results}&tweet.fields=created_at,geo,text&expansions=attachments.poll_ids,attachments.media_keys,author_id,geo.place_id,in_reply_to_user_id,referenced_tweets.id,entities.mentions.username,referenced_tweets.id.author_id&place.fields=contained_within,country,country_code,full_name,geo,id,name&user.fields=created_at,description,entities,id,location,name,pinned_tweet_id,profile_image_url,protected,public_metrics,url,username,verified,withheld"
    payload = {}
    headers = {
        'Authorization': f'Bearer {search_api_bearer}',
        'Cookie': 'guest_id=v1%3A166033111819561959; guest_id_ads=v1%3A166033111819561959; guest_id_marketing=v1%3A166033111819561959; personalization_id="v1_PKCeHmYzBzlDXBwYR96qjg=="'
    }
    response = requests.request("GET", url, headers=headers, data=payload).json()
    print("Successful tweet extraction")
    return response

def load_tweets(keyword, start_date, end_date, results = 500, first_run=True):
    """Pulls tweets from research project API v2
    :param keyword: keyword of tweet for API query
    :param start_date: start date of query, YYYY-MM-DD format, string
    :param end_date: end date of query,  YYYY-MM-DD format, string
    :param results: number of results to return, max 500, int
    :return: loads tweet data to DB"""
    import pandas as pd
    import warnings
    import re
    warnings.filterwarnings("ignore")

    json_object = call_tweets(keyword, start_date, end_date, results)

    # Load tweet text
    df_data = pd.json_normalize(json_object['data'], max_level=5)
    df_text = df_data[['id', 'text', 'created_at', 'author_id']]
    df_text.rename(columns={'id': 'TWEET_ID', 'text': 'TWEET_TEXT', 'created_at': 'CREATED', 'author_id': 'AUTHOR_ID'},
                   inplace=True)
    print("Cleaning tweets..")
    df_text = clean_tweets(df_text)
    print("Tweets cleaned")

    lemmatize(df_text)
    df_text = df_text[['TWEET_ID', 'AUTHOR_ID', 'CREATED', 'TIDY_TWEET', 'LEMM', 'HASHTAGS']]
    df_text['CREATED'] = df_text['CREATED'].astype('datetime64[ns]').dt.date
    df_text['TIDY_TWEET'] = [re.sub("[']", "", item) for item in df_text['TIDY_TWEET']]
    column_list = list(df_text.columns)

    cnx = d.connect_to_database()
    print("Connection established with database")

    for ind, row in df_text.iterrows():
        try:
            try:
                query = (f"""
                INSERT INTO TWEET_TEXT (TWEET_ID, AUTHOR_ID, CREATED, SEARCH_TERM, TIDY_TWEET, LEMM, HASHTAGS)
                VALUES (
                "{row[column_list[0]]}"
                ,"{row[column_list[1]]}"
                ,"{row[column_list[2]]}"
                ,"{keyword}"
                ,"{row[column_list[3]]}"
                ,"{row[column_list[4]]}"
                ,"{row[column_list[5]]}"
                );
                """)
                cnx.execute(query)
            except:
                query = (f"""
                 INSERT INTO TWEET_TEXT (TWEET_ID, AUTHOR_ID, CREATED, SEARCH_TERM, TIDY_TWEET, LEMM, HASHTAGS)
                 VALUES (
                 '{row[column_list[0]]}'
                 ,'{row[column_list[1]]}'
                 ,'{row[column_list[2]]}'
                 ,'{keyword}'
                 ,'{row[column_list[3]]}'
                 ,'{row[column_list[4]]}'
                 ,'{row[column_list[5]]}'
                 );
                 """)
                cnx.execute(query)
        except:
            continue
    print("Data table 'tweet_text' loaded")

    # loading users
    df_author = pd.json_normalize(json_object['includes']['users'], max_level=2)
    df_author = df_author[['id', 'created_at', 'location', 'public_metrics.followers_count',
                           'public_metrics.following_count', 'public_metrics.listed_count',
                           'public_metrics.tweet_count', 'verified']]
    df_author.rename(
        columns={'id': 'AUTHOR_ID', 'created_at': 'CREATED_AT', 'location': 'LOCATION',
                 'public_metrics.followers_count': 'FOLLOWERS_COUNT', \
                 'public_metrics.following_count': 'FOLLOWING_COUNT', 'public_metrics.listed_count': 'LISTED_COUNT',
                 'public_metrics.tweet_count': 'TWEET_COUNT', 'verified': 'VERIFIED'}, inplace=True)

    column_list = list(df_author.columns)

    query_users = "SELECT DISTINCT AUTHOR_ID FROM TWEET_TEXT"
    users = pd.read_sql(query_users, cnx)
    user_list = list(users['AUTHOR_ID'])

    df_author = df_author[df_author['AUTHOR_ID'].isin(user_list)]
    df_author['CREATED_AT'] = df_author['CREATED_AT'].astype('datetime64[ns]').dt.date

    for ind, row in df_author.iterrows():
        try:
            try:
                query = (f"""
                             INSERT INTO USERS
                             VALUES (
                             '{row[column_list[0]]}'
                             ,'{row[column_list[1]]}'
                             ,'{row[column_list[2]]}'
                             ,'{row[column_list[3]]}'
                             ,'{row[column_list[4]]}'
                             ,'{row[column_list[5]]}'
                             ,'{row[column_list[6]]}'
                             ,'{row[column_list[7]]}'
                             );
                             """)
                cnx.execute(query)
            except:
                query = (f"""
                             INSERT INTO USERS
                             VALUES (
                             "{row[column_list[0]]}"
                             ,"{row[column_list[1]]}"
                             ,"{row[column_list[2]]}"
                             ,"{row[column_list[3]]}"
                             ,"{row[column_list[4]]}"
                             ,"{row[column_list[5]]}"
                             ,"{row[column_list[6]]}"
                             ,"{row[column_list[7]]}"
                             );
                             """)
                cnx.execute(query)
        except:
            continue
    print("Data table 'users' loaded")

    #loading user state id
    query = "SELECT STATE, STATE_ABBR, STATE_ID FROM US_STATES"
    results = pd.read_sql_query(query, cnx)

    query_users = "SELECT * FROM USERS"
    users = pd.read_sql(query_users, cnx)

    state_name_dict = dict(zip(results['STATE'], results['STATE_ID']))
    state_abbr_dict = dict(zip(results['STATE_ABBR'], results['STATE_ID']))
    for key, value in state_abbr_dict.items():
        state_name_dict[key] = value

    users[["LOCATION_CLEAN"]] = users[["LOCATION"]].replace(',\ USA$', '', regex=True)
    users[["LOCATION_CLEAN"]] = users[["LOCATION_CLEAN"]].replace('[.]', '', regex=True)
    users[["LOCATION_CLEAN"]] = users[["LOCATION_CLEAN"]].replace('.*,\ (?=[A-Z]{2}$)', '', regex=True)
    users[["LOCATION_CLEAN"]] = users[["LOCATION_CLEAN"]].replace('.*,\ (?=[A-Z]{2}$)', '', regex=True)
    users[["LOCATION_CLEAN"]] = users[["LOCATION_CLEAN"]].replace('.*\ (?=[A-Z]{2}$)', '', regex=True)
    users[["LOCATION_CLEAN"]] = users[["LOCATION_CLEAN"]].replace('.*,\ ', '', regex=True)
    users[["LOCATION_CLEAN"]] = users[["LOCATION_CLEAN"]].replace('.*,(?=[A-Z]{2}$)', '', regex=True)
    users['LOCATION_CLEAN'] = users['LOCATION_CLEAN'].str.upper()
    users['STATE_ID'] = users['LOCATION_CLEAN'].apply(lambda x: [v for k, v in state_name_dict.items() if x == k])
    users['STATE_ID'] = users['STATE_ID'].astype(str)
    users['STATE_ID'] = users['STATE_ID'].replace('[[]', '', regex=True)
    users['STATE_ID'] = users['STATE_ID'].replace('[]]', '', regex=True)
    users = users[users['STATE_ID'].str.contains('\d+')]
    users = users[['AUTHOR_ID', 'STATE_ID']]
    column_list = list(users.columns)

    for ind, row in users.iterrows():
        try:
            query = (f"""
                        INSERT INTO AUTHOR_LOCATION (AUTHOR_ID, STATE_ID)
                        VALUES (
                        "{row[column_list[0]]}"
                        ,"{row[column_list[1]]}"
                        );
                        """)
            cnx.execute(query)
        except:
            continue
    print("Data table 'author_location' loaded")
    cnx.close()
    print("Load process complete")

def database_load(search_term):
    """
    Full ETL process of loading initial search term in databse, extracting related terms and loading these to predict sentiment and time to live
    :param search_term: term used to search and predict sentiment/time to live
    :return: No objects returned, database is loaded
    """
    import datetime as dt
    from tqdm import tqdm
    import os
    import pandas as pd

    try:
        os.mkdir("output_data/")
        print("Created output directory")
    except FileExistsError:
        pass

    print(f'Resetting database new search: "{search_term}" on {(dt.datetime.now()+dt.timedelta(days=-1)).strftime("%Y-%m-%d")}')
    reset_mysql_database()
    load_tweets(search_term, (dt.datetime.now()+dt.timedelta(days=-1)).strftime("%Y-%m-%d"), dt.datetime.now().strftime("%Y-%m-%d"), 500)
    print("Evaluating associated terms...")
    results = gridsearch(search_term)
    results.insert(0,search_term)

    for term in tqdm(results):
        for i in tqdm(range(1, 26)):
            start_date = dt.datetime.now()+dt.timedelta(days=-i-1)
            start_date = start_date.strftime('%Y-%m-%d')
            end_date = dt.datetime.now()+dt.timedelta(days=-i)
            end_date = end_date.strftime('%Y-%m-%d')
            try:
                load_tweets(term, start_date, end_date, 500)
            except:
                print(f"There were no tweets for {term} on {start_date}")

    print('Performing sentiment analysis...')
    print('zip whizz beep bip zip zam')

    analyze_tweets()

    print("Creating supervised model and predicting sentiments...")
    create_sentiment_model()

    cnx = connect_to_database()
    query1 = f"""
    SELECT
    CREATED,
    COUNT(TWEET_ID) AS TWEET_COUNT,
    SEARCH_TERM,
    CASE WHEN SEARCH_TERM = '{search_term}' then 1 else 0 end as primary_search_term
    FROM TWEET_TEXT
    GROUP BY CREATED, SEARCH_TERM
    ;
    """
    df1 = pd.read_sql_query(query1,cnx)

    query2 = """
    SELECT
    R.REGION
    ,D.DIVISION
    ,S.STATE_ABBR
    ,COUNT(A.AUTHOR_ID) AS AUTHOR_COUNT
    FROM AUTHOR_LOCATION A
    LEFT JOIN US_STATES S ON S.STATE_ID = A.STATE_ID
    LEFT JOIN DIVISIONS D ON D.DIV_ID = S.DIV_ID
    LEFT JOIN REGIONS R ON R.REG_ID = D.REG_ID
    GROUP BY R.REGION, D.DIVISION, S.STATE_ABBR
    ;
    """
    df2 = pd.read_sql_query(query2, cnx)

    query3 = """
        SELECT
        SEARCH_TERM,
        OVERALL_EMO
        FROM TWEET_TEXT
        GROUP BY SEARCH_TERM, OVERALL_EMO
        ;
        """
    df3 = pd.read_sql_query(query3, cnx)

    writer = pd.ExcelWriter('output_data/load_process_metrics.xlsx', engine='xlsxwriter')
    df1.to_excel(writer, sheet_name='SEARCH_METRICS')
    df2.to_excel(writer, sheet_name='LOCATION_METRICS')
    df3.to_excel(writer, sheet_name='SENTIMENT_METRICS')
    writer.save()
    writer.close()

    print("Load Process Complete")

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
    import pandas as pd
    from tqdm import tqdm

    cnx = connect_to_database()
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

    cnx = connect_to_database()
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
    from itertools import product
    from collections import Counter

    cnx = connect_to_database()
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

        kw_list = get_associated_keywords(df, search_term, perc_in_words=perc_val.astype(float),
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
    from sklearn.metrics import accuracy_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GridSearchCV, KFold
    from tqdm import tqdm

    cnx = connect_to_database()

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

    cnx = connect_to_database()
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

def generate_report():
    """
    Generates report of tweet analysis.
    :return: None, html file generates to working directory as Sentiment_Report.html
    """
    import warnings
    warnings.simplefilter(action='ignore') #added due to known bug in current pandas version
    import os
    import numpy as np
    import pandas as pd

    print("Generating charts...")
    streamgraph()
    hashtag_chart()
    emo_choropleth()
    forecast_chart()
    interactive_tweet_trends()
    animated_emo_choropleth()
    emotion_by_div_reg()
    simple_trend()
    division_author_count()
    print("Charts created")
    print("Compiling report")
    cnx = connect_to_database()
    db_return1 = cnx.execute("""
    SELECT DISTINCT 
    SEARCH_TERM 
    ,MAX(CREATED) AS END_DATE
    ,MIN(CREATED) AS START_DATE
    FROM TWEET_TEXT
    GROUP BY SEARCH_TERM
    ORDER BY END_DATE DESC
    LIMIT 1
    """)
    results = db_return1.fetchall()

    db_return2 = cnx.execute("""
        SELECT COUNT(TWEET_ID) FROM TWEET_TEXT
        """)
    results2 = db_return2.fetchall()

    db_return3 = cnx.execute("""
    SELECT
    ROUND(AVG(AVG_T)) AS AVG_TWEET_P_DAY
    FROM (
        SELECT DISTINCT 
        COUNT(TWEET_ID)/COUNT(DISTINCT SEARCH_TERM) AS AVG_T
        FROM TWEET_TEXT
        GROUP BY CREATED) X """)

    results3 = db_return3.fetchall()

    db_return4 = cnx.execute("""
        SELECT
        COUNT(OVERALL_EMO) AS OE_COUNT
        ,OVERALL_EMO
        FROM TWEET_TEXT
        GROUP BY OVERALL_EMO
        ORDER BY COUNT(OVERALL_EMO) DESC""")

    results4 = db_return4.fetchall()
    results4.sort(reverse=True)
    result4_vals = [x[0] for x in results4]
    result4_mean = round(sum(result4_vals)/len(result4_vals),2)

    db_return5 = cnx.execute("""
        SELECT DISTINCT 
        SEARCH_TERM 
        FROM TWEET_TEXT
        """)
    results5 = db_return5.fetchall()

    df_trend = pd.read_sql_query("""
    SELECT DISTINCT 
            CREATED, COUNT(TWEET_ID)/COUNT(DISTINCT SEARCH_TERM) AS AVG_T
            FROM TWEET_TEXT
            GROUP BY CREATED""", cnx)

    slope, intercept = np.polyfit(x=df_trend.index,y=df_trend['AVG_T'], deg=1, rcond=None, full=False, w=None, cov=False)

    df_trend['AVG_T_SHIFT'] = df_trend['AVG_T'].shift(1)
    df_trend['difference'] = df_trend['AVG_T'] - df_trend['AVG_T_SHIFT']
    df_trend['difference'] = df_trend['difference'].abs()

    trend = None
    trend_term = None
    if slope > 0:
        trend = "Growing"
        trend_term = "Gaining"
    elif slope < 0:
        trend = "Shrinking"
        trend_term = "Losing"
        slope = slope*-1
    else:
        trend = "Flat"
        trend_term = "Is Stable With"



    f = open('Sentiment_Report.html', 'w')

    html_template = f"""
    <h1>TweetERA Report</h1>
    <p>
    Original Search Term: {results[0][0]}
    <br>
    All Search Terms: {', '.join([x[0] for x in results5])}
    <br>
    Start Date: {results[0][2]}
    <br>
    End Date: {results[0][1]}
    <br>
    Total Tweets Obtained: {results2[0][0]}
    <br>
    Average Tweets per Day: {results3[0][0]}
    <br>
    <br>
    Overall, users felt mostly <b>{results4[0][1]}</b> about the topic, with a total tweet count of {results4[0][0]}. 
    <br>
    With an average tweet count by sentiment being {result4_mean}, this is <i>{round(results4[0][0]/result4_mean,4)*100}%</i> above the mean.
    <br>
    <br>
    </p>
    <iframe src="output_data/streamgraph.html" width="1000" height="650" frameBorder="0">></iframe>
    <br>
    <h2>Tweet Trends and Forecast</h2>
    <br>
    <iframe src="output_data/simple_trend.html" width="950" height="300" frameBorder="0">></iframe>
    <br>
    The trend analysis of this search term indicates volume is <b>{trend}</b>. This trend is <b>{trend_term}</b> approximately <i>{round(slope,2)}</i> tweets per day.
    <br>
    <br>
    The following chart can be used to see how much a trend is growing or shrinking. The top half shows the average tweet volume per day, and the 
    <br>
    bottom shows the sentiment trend. Selecting a segment on the top will allow you to focus on a section below. Selecting an emotion will 
    <br>
    filter all others. Press F5 or reload to reset the chart. This chart can help isolate spikes in emotions based on Twitter users' exposure to new information 
    <br>
    in the timeline.
    <br> 
    <iframe src="output_data/interactive_tweet_trends.html" width="950" height="700" frameBorder="0">></iframe>
    <br>
    <br>
    <br>
    The following chart shows a prediction of tweets over the next 10 days. The black line indicates the current average volume, while the 
    <br>
    magenta line indicates the predictions. If a search term has little volume, predictions can be hard to make and should be evaluated in context of the tweet volume.
    <br>
    <iframe src="output_data/forecast_chart.html" width="950" height="700" frameBorder="0">></iframe>
    <br>
    <br>
    <h2> Hashtags </h2>
    This chart shows the hashtag use by volume. A hashtag-driven topic is immediately popular at a particular time; these may provide 
    <br>
    good alternate search terms.
    <br> 
    <iframe src="output_data/hashtag_chart.html" width="850" height="400" frameBorder="0">></iframe>
    <br>
    <h2> User Locations </h2>
    <br>
    This data represents users who entered enough location data to identify their home state and does not reflect the location of a user 
    <br>
    at the time they created a tweet. You will notice that tweet volume is lower in this section. Lower tweet volume by location is 
    <br>
    dependent on the number of users with a set location.
    <br>
    <iframe src="output_data/emo_choropleth.html" width="1000" height="600" frameBorder="0">></iframe>
    <br>
    <iframe src="output_data/animated_emo_choropleth.html" width="1000" height="600" frameBorder="0">></iframe>
    <br>
    <br>
    <iframe src="output_data/division_author_count.html" width="950" height="400" frameBorder="0">></iframe>
    <br>
    <br>
    <iframe src="output_data/emotion_by_div_reg.html" width="1200" height="2000" frameBorder="0">></iframe>
    """

    f.write(html_template)
    f.close()

    print(f"Report saved to: {os.getcwd()}\Sentiment_Report.html")

def check_trend(*args):
    """
    Uses google trend to build a simple line chart of the current trend by keyword/phrase
    :param keyword: keyword or phrase, or many keywords/phrases separated by commas. Must be strings.
    :return: creates a plotly image which generates from .show()
    """
    from pytrends.request import TrendReq
    pytrends = TrendReq(hl='en-US', tz=360)
    kw_list = [*args]
    pytrends.build_payload(kw_list, cat=0, timeframe='today 12-m')
    data = pytrends.interest_over_time()
    data = data.reset_index()

    import plotly.express as px
    fig = px.line(data, x="date", y=kw_list, title='Keyword Web Search Interest Over Time')
    fig.show()
    
    
def streamgraph():
    '''
    Creates a streamgraph: counts of overall emotions by date
    :return: altair streamgraph visualization
    '''
    import altair as alt
    from altair_saver import save
    import pandas as pd
    import numpy as np

    cnx = connect_to_database()

    query = 'SELECT * FROM TWEET_TEXT;'
    df = pd.read_sql_query(query, cnx)

    alt.data_transformers.disable_max_rows()
    
    colors = ['#a8201a', '#ec9a29', '#7e935b', '#143642', '#857f83', '#526797', '#0f8b8d']
    emos = [ 'Angry', 'Fear', 'Happy', 'Mixed', 'Neutral', 'Sad', 'Surprise']

    chart = alt.Chart(df, title=f"Search Terms: {np.unique(df['SEARCH_TERM'])}").mark_area().encode(
        alt.X('CREATED:T',
            axis=alt.Axis(domain=False, grid=False, tickSize=0)
        ),
        alt.Y('count(OVERALL_EMO):N', stack='center',
             axis=None, title="", ),
        alt.Color('OVERALL_EMO:N',
            scale=alt.Scale(domain=emos,
                            range=colors),
        )
    ).properties(height=500, width=800).configure_view(strokeOpacity=0)

    save(chart, "output_data/streamgraph.html")


def emo_choropleth():
    '''
    Creates a choropleth map of most common overall_emo by state, excluding 'Neutral' and 'Mixed'
    :return: None, image saved to "output_data" directory.
    '''
    import pandas as pd
    import plotly.express as px
    import numpy as np
    from collections import Counter

    cnx = connect_to_database()

    query = """
    SELECT * FROM TWEET_TEXT
    JOIN AUTHOR_LOCATION
    USING (AUTHOR_ID)
    JOIN US_STATES
    USING (STATE_ID)"""
    df = pd.read_sql_query(query, cnx)

    colors = {'Angry':'#a8201a', 'Fear':'#ec9a29', 'Happy':'#7e935b', 'Mixed':'#143642',
              'Neutral':'#857f83', 'Sad':'#526797', 'Surprise':'#0f8b8d'}
    
    most_common_list = []
    for state in df['STATE_ABBR']:
        state_df = df.where(df.STATE_ABBR == state).dropna()
        counter = Counter(state_df['OVERALL_EMO'])
        try:
            if counter.most_common()[0][1] == counter.most_common()[1][1]:
                most_common_list.append('Mixed')
            else:
                most_common_list.append(counter.most_common()[0][0])
        except IndexError:
            most_common_list.append(counter.most_common()[0][0])

    df['MOST_COMMON_EMO'] = most_common_list
    df = df.sort_values('MOST_COMMON_EMO')

    fig = px.choropleth(df,
                        locations='STATE_ABBR', 
                        locationmode="USA-states", 
                        scope="usa",
                        color='MOST_COMMON_EMO',
                        color_discrete_map=colors,
                        )

    fig.update_layout(
          title_text = f"Overall Emotion by State (of users with location listed) <br> Search Terms: {df['SEARCH_TERM'].unique()}",
          title_font_size = 14,
          title_font_color="black", 
          title_x=0.45, 
             )
    fig.write_html('output_data/emo_choropleth.html')


def hashtag_chart():
    '''
    Creates a bar chart with top 10(max) hashtags
    :return: altair bar chart visualization
    '''
    import altair as alt
    import pandas as pd
    import numpy as np
    from altair_saver import save
    from collections import Counter
    
    cnx = connect_to_database()

    query = 'SELECT * FROM TWEET_TEXT;'
    df = pd.read_sql_query(query, cnx)
    

    hash_counts = Counter(df['HASHTAGS'])
    hash_counts.pop('[]')
    
    hash_df = pd.DataFrame.from_dict(hash_counts, orient='index', columns=['count'])
    hash_df = hash_df.reset_index()
    hash_df = hash_df.sort_values('count', ascending=False)
    hash_df = hash_df[:10]
    hash_df.reset_index(inplace=True, drop=True)


    bars = alt.Chart(hash_df, title=["Top Hashtags", f"Search Terms: {df['SEARCH_TERM'].unique()}"]).mark_bar(color = '#2182bd').encode(
        y = alt.Y('index:N', sort='-x', axis=alt.Axis(grid=False, title='Hashtag')),
        x = alt.X('count:Q', axis=alt.Axis(grid=True, title="Tweet Count"))
    ).properties(height=300, width=500).configure_axis(
    labelFontSize=14)
    
    save(bars, "output_data/hashtag_chart.html")
    
def forecast_chart():
    '''
    Creates a line chart with projected tweet volumes for next 10 days
    :return: altair bar chart visualization
    '''

    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMA
    import datetime
    import altair as alt
    from altair_saver import save

    cnx = connect_to_database()

    query = """
    SELECT COUNT(TWEET_ID)/COUNT(DISTINCT SEARCH_TERM) AS COUNT, 
    CREATED 
    FROM TWEET_TEXT
    GROUP BY CREATED"""
    df = pd.read_sql_query(query, cnx)
    df.drop(df.tail(1).index,inplace=True)
    x_dates = []
    df['CREATED']= pd.to_datetime(df['CREATED'])
    for i in range(10):
        x_dates.append(df['CREATED'].max() + datetime.timedelta(days=i+1))

    ARIMAmodel = ARIMA(df['COUNT'], order = (2, 0, 2))
    ARIMAmodel = ARIMAmodel.fit()
    y_pred = ARIMAmodel.get_forecast(10)
    y_pred_df = y_pred.conf_int(alpha = 0.05)
    y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
    y_pred_df['CREATED'] = x_dates
    y_pred_df.loc[len(y_pred_df)] = [None,None,df.iat[-1, 0], df.iat[-1,1]]

    df = pd.concat([df,y_pred_df])
    lines = alt.Chart(df).mark_line().encode(
    x=alt.X('CREATED',axis=alt.Axis(grid=False)),
    y = alt.Y('COUNT'),
    y2 = alt.Y('Predictions:Q')
    )
    plot_title = alt.TitleParams("Historical and Predicted Tweet Counts", subtitle=["Ten day ARIMA prediction of tweet volumes"])
    base = alt.Chart(df.reset_index(), title = plot_title).encode(alt.X('CREATED', title = "Tweet Date"))

    lines = alt.layer(
        base.mark_line(color='black').encode(alt.Y('COUNT')),
        base.mark_line(color='#E8175D').encode(alt.Y('Predictions:Q', title = "Average Tweet Count"))
    ).properties(height = 500, width = 800)
    save(lines, "output_data/forecast_chart.html")

def interactive_tweet_trends():
    """
    Creates interactive chart of tweet trends and emotions over time.
    :return: None, chart is saved to output folder
    """
    import pandas as pd
    import altair as alt
    from altair_saver import save

    cnx = connect_to_database()

    df_volume = pd.read_sql_query("""
    SELECT
    created
    ,round(count(tweet_id)/count(distinct search_term)) as avg_vol
    FROM TWEET_TEXT
    group by created;
    """, cnx)

    df_sent = pd.read_sql_query("""
    select 
    created
    ,overall_emo
    ,count(overall_emo)/count(distinct search_term) as emo_count
    from tweet_text
    group by created, overall_emo;
    """, cnx)

    alph_colors = ['#a8201a', '#ec9a29', '#7e935b', '#143642', '#857f83', '#526797', '#0f8b8d']
    
    brush = alt.selection_interval(encodings=['x'])
    colorConditionDC = alt.condition(brush, alt.value('#2182bd'), alt.value('gray'))

    volume = alt.Chart(df_volume).mark_bar(color='grey').encode(
        x=alt.X('created', title="Date Created"),
        y=alt.Y('avg_vol:Q', title="Average Tweet Volume")
    ).properties(height=200, width=700)

    i_volume = volume.add_selection(brush).encode(color=colorConditionDC).resolve_scale(y='shared'
                                                                                        )
    sent_line = alt.Chart(df_sent).mark_line(size=2).encode(
        x=alt.X('created', title="Date Created"),
        y=alt.Y('emo_count:Q', title="Count of Emotions"),
        color=alt.Color('overall_emo', legend=None),
        tooltip='overall_emo'
    ).properties(height=200, width=700)

    selection = alt.selection_multi(fields=['overall_emo'])
    make_selector = alt.Chart(df_sent).mark_rect().encode(
        y=alt.Y('overall_emo', title=None),
        color='overall_emo'
    ).add_selection(selection).properties(title="Click to Filter")

    i_sent_line = sent_line.transform_filter(brush).resolve_scale(y='shared').transform_filter(selection)

    out = (i_volume & (i_sent_line | make_selector)).configure_range(
        category=alt.RangeScheme(alph_colors)
    ).properties(
        title={
            "text": ["Tweet Volume and Sentiment Over Time"],
            "subtitle": ["",
                         "I'm Interactive! Select a section on the bar chart to zoom in on sentiment values.",
                         "Select an emotion to focus.", ""]}
    )

    save(out, "output_data/interactive_tweet_trends.html")

def animated_emo_choropleth():
    """
    Creates an animated choropleth that plays frames from start date to end date.
    :return: None, html files saved to output_data
    """
    import pandas as pd
    from collections import Counter
    import plotly.express as px

    cnx = connect_to_database()

    df = pd.read_sql_query("""
    SELECT * FROM TWEET_TEXT
    JOIN AUTHOR_LOCATION
    USING (AUTHOR_ID)
    JOIN US_STATES
    USING (STATE_ID)""", cnx)

    most_common_list = []
    for state in df['STATE_ABBR']:
        state_df = df.where(df.STATE_ABBR == state).dropna()
        counter = Counter(state_df['OVERALL_EMO'])
        try:
            if counter.most_common()[0][1] == counter.most_common()[1][1]:
                most_common_list.append('Mixed')
            else:
                most_common_list.append(counter.most_common()[0][0])
        except IndexError:
            most_common_list.append(counter.most_common()[0][0])

    df['MOST_COMMON_EMO'] = most_common_list
    df = df.sort_values('MOST_COMMON_EMO')
    
    colors = {'Angry':'#a8201a', 'Fear':'#ec9a29', 'Happy':'#7e935b', 'Mixed':'#143642',
              'Neutral':'#857f83', 'Sad':'#526797', 'Surprise':'#0f8b8d'}
        
    fig = px.choropleth(df,
                        locations='STATE_ABBR',
                        locationmode="USA-states",
                        color='MOST_COMMON_EMO',
                        color_discrete_map=colors,
                        scope="usa",
                        animation_frame='CREATED')
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000

    fig.write_html('output_data/animated_emo_choropleth.html')

def emotion_by_div_reg():
    import pandas as pd
    import altair as alt
    import numpy as np
    from altair_saver import save
    from sklearn import preprocessing

    cnx = connect_to_database()
    df = pd.read_sql_query("""
    SELECT 
    COUNT(T.TWEET_ID) AS COUNT,
    T.OVERALL_EMO,
    D.DIVISION,
    R.REGION
    FROM TWEET_TEXT T 
    JOIN AUTHOR_LOCATION A ON T.AUTHOR_ID = A.AUTHOR_ID
    JOIN US_STATES S ON A.STATE_ID = S.STATE_ID
    JOIN DIVISIONS D ON D.DIV_ID = S.DIV_ID
    JOIN REGIONS R ON R.REG_ID = D.REG_ID
    GROUP BY T.OVERALL_EMO, D.DIVISION, R.REGION
    """, cnx)

    df['count_norm'] = preprocessing.normalize(np.array(df['COUNT']).reshape(-1, 1), axis=0)
    div_charts = []
    reg_charts = []
    
    colors = ['#a8201a', '#ec9a29', '#7e935b', '#143642', '#857f83', '#526797', '#0f8b8d']
    emos = [ 'Angry', 'Fear', 'Happy', 'Mixed', 'Neutral', 'Sad', 'Surprise']

    for e in df['OVERALL_EMO'].unique():
        df_div = df[df['OVERALL_EMO'] == e]
        divisions = alt.Chart(df_div).mark_bar().encode(
            y=alt.Y('DIVISION', title=''),
            x=alt.X('count_norm:Q', title='Tweet Count', scale=alt.Scale(domain=[0, 1])),
            color = alt.Color('OVERALL_EMO:N',
                  scale=alt.Scale(domain=emos,
                                  range=colors)),
        ).properties(title=f'{e}')
        div_charts.append(divisions)

    for e in df['OVERALL_EMO'].unique():
        df_div = df[df['OVERALL_EMO'] == e]
        regions = alt.Chart(df_div).mark_bar().encode(
            y=alt.Y('REGION', title=''),
            x=alt.X('count_norm:Q', title='', scale=alt.Scale(domain=[0, 1])),
            color = alt.Color('OVERALL_EMO:N',
                  scale=alt.Scale(domain=emos,
                                  range=colors)),
        ).properties(title=f'{e}')
        reg_charts.append(regions)

    divisions = alt.vconcat(*div_charts).properties(title="Sentiment by US Division")
    regions = alt.vconcat(*reg_charts).properties(title="Sentiment by US Region")

    chart = alt.hconcat(divisions, regions).resolve_axis(
        x='independent',
        y='independent',
    ).properties(title={"text": "Tweet Sentiment Count by US Division and Location",
                        "subtitle": ["Tweet counts normalized to adjust for variation in daily volume", " "]})

    save(chart, "output_data/emotion_by_div_reg.html")

def simple_trend():
    import pandas as pd
    import altair as alt
    from altair_saver import save

    cnx = connect_to_database()
    query = """
    SELECT  
        CREATED,
        COUNT(TWEET_ID)/COUNT(DISTINCT SEARCH_TERM) AS AVG_T
        FROM TWEET_TEXT
        GROUP BY CREATED
    """
    df = pd.read_sql_query(query, cnx)

    base = alt.Chart(df).mark_point().encode(
        x=alt.X('CREATED:T', axis=alt.Axis(grid=False, title="Date Created")),
        y=alt.Y('AVG_T:Q', axis=alt.Axis(grid=True, title="Average Tweet Volume"))
    )

    chart = base+base.transform_regression('CREATED', 'AVG_T').mark_line(color="red")

    save(chart.properties(height=200, width=700), "output_data/simple_trend.html")

def division_author_count():
    import pandas as pd
    import altair as alt
    from altair_saver import save

    cnx = connect_to_database()

    query = """
    SELECT 
    COUNT(A.AUTHOR_ID) AS ACOUNT,
    D.DIVISION
    FROM AUTHOR_LOCATION A 
    JOIN US_STATES S ON A.STATE_ID = S.STATE_ID
    JOIN DIVISIONS D ON S.DIV_ID = D.DIV_ID
    GROUP BY D.DIVISION
    ORDER BY COUNT(A.AUTHOR_ID) DESC
    """

    df = pd.read_sql_query(query, cnx)

    chart = alt.Chart(df, title="Authors with Set Locations").mark_bar(color='#2182bd').encode(
        y=alt.Y('DIVISION:N', axis=alt.Axis(grid=False, title='US Division'), sort='-x'),
        x=alt.X('ACOUNT:Q', title="Author Count")
    ).properties(height=300, width=600)

    save(chart, "output_data/division_author_count.html")

