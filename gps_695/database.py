import numpy as np


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
    from gps_695 import database as d
    # import mysql
    # import sqlalchemy
    # from mysql.connector import connect, Error

    file = open('gps_695/database_table_creation.sql', 'r')
    sql = file.read()
    file.close

    cnx = d.connect_to_database()
    cnx.execute(sql)
    cnx.close()

def reset_mysql_database():
    """
    Function resets mysql database for new data loading. The process will remove all tables from the database and recreate it.
    :return: None
    """
    from gps_695 import database as d

    file = open('gps_695/database_clear.sql', 'r')
    sql = file.read()
    file.close

    cnx = d.connect_to_database()
    cnx.execute(sql)
    cnx.close()
    d.create_mysql_database()
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
    from gps_695 import database as d
    from gps_695 import nlp as n
    import warnings
    import re
    warnings.filterwarnings("ignore")

    json_object = d.call_tweets(keyword, start_date, end_date, results)

    # Load tweet text
    df_data = pd.json_normalize(json_object['data'], max_level=5)
    df_text = df_data[['id', 'text', 'created_at', 'author_id']]
    df_text.rename(columns={'id': 'TWEET_ID', 'text': 'TWEET_TEXT', 'created_at': 'CREATED', 'author_id': 'AUTHOR_ID'},
                   inplace=True)
    print("Cleaning tweets..")
    df_text = n.clean_tweets(df_text)
    print("Tweets cleaned")

    n.lemmatize(df_text)
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
    from gps_695 import database as d
    from gps_695 import nlp as n
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
    d.reset_mysql_database()
    d.load_tweets(search_term, (dt.datetime.now()+dt.timedelta(days=-1)).strftime("%Y-%m-%d"), dt.datetime.now().strftime("%Y-%m-%d"), 500)
    print("Evaluating associated terms...")
    results = n.gridsearch(search_term)
    results.insert(0,search_term)

    for term in tqdm(results):
        for i in tqdm(range(1, 26)):
            start_date = dt.datetime.now()+dt.timedelta(days=-i-1)
            start_date = start_date.strftime('%Y-%m-%d')
            end_date = dt.datetime.now()+dt.timedelta(days=-i)
            end_date = end_date.strftime('%Y-%m-%d')
            try:
                d.load_tweets(term, start_date, end_date, 500)
            except:
                print(f"There were no tweets for {term} on {start_date}")

    print('Performing sentiment analysis...')
    print('zip whizz beep bip zip zam')

    n.analyze_tweets()

    print("Creating supervised model and predicting sentiments...")
    n.create_sentiment_model()

    cnx = d.connect_to_database()
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