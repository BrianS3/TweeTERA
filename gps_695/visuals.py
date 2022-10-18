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
    from gps_695 import visuals as v
    from gps_695 import database as d

    print("Generating charts...")
    v.streamgraph()
    v.hashtag_chart()
    v.emo_choropleth()
    v.forecast_chart()
    v.interactive_tweet_trends()
    v.animated_emo_choropleth()
    v.emotion_by_div_reg()
    v.simple_trend()
    v.division_author_count()
    print("Charts created")
    print("Compiling report")
    cnx = d.connect_to_database()
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
    from gps_695 import database as d
    import pandas as pd
    import numpy as np

    cnx = d.connect_to_database()

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
    from gps_695 import database as d
    import pandas as pd
    import plotly.express as px
    import numpy as np
    from collections import Counter

    cnx = d.connect_to_database()

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
    from gps_695 import database as d
    
    cnx = d.connect_to_database()

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
    from gps_695 import database as d
    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMA
    import datetime
    import altair as alt
    from altair_saver import save

    cnx = d.connect_to_database()

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
    from gps_695 import database as d

    cnx = d.connect_to_database()

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
    from gps_695 import database as d

    cnx = d.connect_to_database()

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
    from gps_695 import database as d

    cnx = d.connect_to_database()
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
    from gps_695 import database as d

    cnx = d.connect_to_database()
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
    from gps_695 import database as d

    cnx = d.connect_to_database()

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

