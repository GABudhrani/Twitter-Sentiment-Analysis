import base64
import io
import json

import plotly
from django.shortcuts import render
from django.http import  HttpResponse
import tweepy
import pandas as pd
import numpy as np
import csv
import re
import string
import glob
import random
import requests
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import nltk
from matplotlib_inline.backend_inline import FigureCanvas
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import plotly.express as px
# import chart_studio.tools as cst
from textblob import TextBlob
# import chart_studio.plotly as py
import plotly.offline as pyo
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
import joblib
# import gensim
# from collections import Counter
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# def index(request):
#     context = {'a':"hello world"}
#     return render(request, 'index.html', context)
#     # return HttpResponse({'a':1 })

def tweet_senti(request):
    if request.method == 'POST':
        # print(request.POST.dict())
        keyword = request.POST.get('keyword')
        count = request.POST.get('count')

# keyword = 'covid'
# count = 15
        api_key = 'FBlx4knKrSmS5a8NMEYAyhK1L'
        api_secret = 'BNSn115HQ3xxd3XXT5pJXzo4P9yaRkNTK6CpShvQaOD4uso9l8'
        access_key = '1262253125626327040-rONz8m4AVgRD0JiQVco2AvBULvelZ5'
        access_secret = '34CcxIxRiOx2XjaAQBt9Kj7n1hdgrxcsxeSz4kE377LjA'

        auth = tweepy.OAuthHandler(api_key, api_secret)  # Pass in Consumer key and secret for authentication by API
        auth.set_access_token(access_key, access_secret)  # Pass in Access key and secret for authentication by API
        api = tweepy.API(auth, wait_on_rate_limit=True,
                         wait_on_rate_limit_notify=True)  # Sleeps when API limit is reached
        try:
            api.verify_credentials()
            print("Authentication OK")
        except:
            print("Error during authentication")
        # number_tweets = int(input("Enter Number of tweets: "))
        # topic = input("Enter Topic: ")

        number_tweets = int(count)
        # Specify the Search word
        search_word = keyword

        def get_tweets(search_query, num_tweets):
            tweets = []

            # Using this Function We will collect the tweets
            tweet_list = [tweets for tweets in
                          tweepy.Cursor(api.search, q=search_query, lang="en", tweet_mode='extended').items(num_tweets)]

            # Here we are retriving the tweets from the tweeter database and one by one and storing it in the following variable
            for tweet in tweet_list[::-1]:
                tweet_id = tweet.id
                created_at = tweet.created_at
                text = tweet.full_text
                location = tweet.user.location
                retweet = tweet.retweet_count
                favorite = tweet.favorite_count
                data = [tweet_id, created_at, text, location, retweet, favorite]
                tweets.append(data)

            return tweets

        # This Queary will exclude retweets, Links, replies
        search_query = search_word + " -filter:links AND -filter:retweets AND -filter:replies"
        # with open('Data/covid.csv', encoding='utf-8') as data:
        #     # Retrving the most resent tweet ID
        #     latest_tweet = int(list(csv.reader(data))[-1][0])
        tweets = get_tweets(search_query, number_tweets)

        test = pd.DataFrame(tweets, columns=['tweet_id', 'created_at', 'tweet', 'location', 'retweet', 'favorite'])
        # print(tweets_data)
        test = test.drop_duplicates(subset=['tweet_id'])  # drop duplicate values
        test['location'] = test['location'].fillna('No location')  # Replace "NaN" values with "No Location"

        test['len'] = test['tweet'].str.len()
        test = test.drop_duplicates(subset=['tweet_id'])  # drop duplicate values
        test_corpus = []

        for i in range(0, test.shape[0]):
            review = re.sub('[^a-zA-Z]', ' ', test['tweet'][i])
            review = review.lower()
            review = review.split()

            ps = PorterStemmer()

            # stemming
            review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

            # joining them back with space
            review = ' '.join(review)
            test_corpus.append(review)
        # creating bag of words

        cv = CountVectorizer(max_features=100)
        x_test = cv.fit_transform(test_corpus).toarray()
        # print(x_test.shape)
        sc = StandardScaler()
        sc.fit(x_test)
        x_test = sc.transform(x_test)
        xgboost_model = joblib.load("/Users/girishbudhrani/Desktop/G drive/Projects/Twiter/tp/twitter_webapp/model/xgboost_model")
        # print(f"Predicted value is :- {xgboost_model.predict(x_test)}")
        Sentiment = xgboost_model.predict(x_test)

        test["Sentiment"] = Sentiment
        # print(test)
        json_records = test.reset_index().to_json(orient='records')
        data_final = json.loads(json_records)

        print(test)
        # Defining my NLTK stop words and my user-defined stop words
        stop_words = list(nltk.corpus.stopwords.words('english'))
        user_stop_words = ['2020', 'year', 'many', 'much', 'amp', 'next', 'cant', 'wont', 'hadnt',
                           'havent', 'hasnt', 'isnt', 'shouldnt', 'couldnt', 'wasnt', 'werent',
                           'mustnt', '’', '...', '..', '.', '.....', '....', 'been…', 'one', 'two',
                           'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'aht',
                           've', 'next']
        alphabets = list(string.ascii_lowercase)
        stop_words = stop_words + user_stop_words + alphabets
        word_list = words.words()  # all words in English language
        emojis = list(UNICODE_EMOJI.keys())  # full list of emojis

        # Function to remove punctuations, links, emojis, and stop words
        def preprocessTweets(tweet):
            tweet = tweet.lower()  # has to be in place
            # Remove urls
            tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
            # Remove user @ references and '#' from tweet
            tweet = re.sub(r'\@\w+|\#|\d+', '', tweet)
            # Remove stopwords
            tweet_tokens = word_tokenize(tweet)  # convert string to tokens
            filtered_words = [w for w in tweet_tokens if w not in stop_words]
            filtered_words = [w for w in filtered_words if w not in emojis]
            filtered_words = [w for w in filtered_words if w in word_list]

            # Remove punctuations
            unpunctuated_words = [char for char in filtered_words if char not in string.punctuation]
            unpunctuated_words = ' '.join(unpunctuated_words)

            return "".join(unpunctuated_words)  # join words with a space in between them

        # function to obtain adjectives from tweets
        def getAdjectives(tweet):
            tweet = word_tokenize(tweet)  # convert string to tokens
            tweet = [word for (word, tag) in pos_tag(tweet)
                     if tag == "JJ"]  # pos_tag module in NLTK library
            return " ".join(tweet)  # join words with a space in between them

        test['Processed_Tweets'] = test['tweet'].apply(preprocessTweets)
        test['Tweets_Adjectives'] = test['Processed_Tweets'].apply(getAdjectives)

        # function to return words to their base form using Lemmatizer
        def preprocessTweetsSentiments(tweet):
            tweet_tokens = word_tokenize(tweet)
            lemmatizer = WordNetLemmatizer()  # instatiate an object WordNetLemmatizer Class
            lemma_words = [lemmatizer.lemmatize(w) for w in tweet_tokens]
            return " ".join(lemma_words)

        # Apply preprocessTweetsSentiments function to the 'Processed Tweets' column to generate a new column
        # called 'Processed_Tweets'
        test['Tweets_Sentiments'] = test['Processed_Tweets'].apply(preprocessTweetsSentiments)
        # Combine all words into a list
        tweets_long_string = test['Processed_Tweets'].tolist()
        tweets_list = []
        for item in tweets_long_string:
            item = item.split()
            for i in item:
                tweets_list.append(i)
        # Use the Built-in Python Collections module to determine Word frequency

        counts = Counter(tweets_list)
        data = pd.DataFrame.from_dict(counts, orient='index').reset_index()
        data.columns = ['Words', 'Count']
        data.sort_values(by='Count', ascending=False, inplace=True)

        # This function is helping to obtain a Subjectivity Score
        def getSubjectivity(tweet):
            return TextBlob(tweet).sentiment.subjectivity

        # This function is helping to obtain a Polarity Score
        def getPolarity(tweet):
            return TextBlob(tweet).sentiment.polarity

        # This function is helping to obtain aSentiment category
        def getSentimentTextBlob(polarity):
            if polarity < 0:
                return "Negative"
            elif polarity == 0:
                return "Neutral"
            else:
                return "Positive"

        # Applying the functions and adding them to an new columns
        test['Subjectivity'] = test['Tweets_Sentiments'].apply(getSubjectivity)
        test['Polarity'] = test['Tweets_Sentiments'].apply(getPolarity)
        # test['Sentiment'] = test['Polarity'].apply(getSentimentTextBlob)
        # print(test['Sentiment'].value_counts())
        bar_chart = test['Sentiment'].value_counts().rename_axis('Sentiment').to_frame(
            'Total Tweets').reset_index()
        sentiments_barchart = px.bar(bar_chart, x='Sentiment', y='Total Tweets', color='Sentiment')

        # sentiments_barchart.update_layout(title='Distribution of Sentiments Results',margin={"r": 0, "t": 30, "l": 0, "b": 0})

        barplot_senti = plotly.offline.plot(sentiments_barchart, auto_open=False, output_type="div")

        colors = ['rgb(8,48,107)', 'rgb(8,81,156)', 'rgb(33,113,181)', 'rgb(66,146,198)',
                  'rgb(107,174,214)', 'rgb(158,202,225)', 'rgb(198,219,239)',
                  'rgb(222,235,247)', 'rgb(247,251,255)', 'rgb(247,253,255)']

        # Set layout for Plotly Subplots
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "domain"}]],
                            vertical_spacing=0.001)

        # Add First Plot
        # print(data)
        fig.add_trace(go.Bar(x=data['Count'].head(10), y=data['Words'].head(10), marker=dict(color='rgba(66,146,198, 1)',line=dict(color='Black'), ),name='Bar Chart', orientation='h'), 1, 1)

        # Add Second Plot
        fig.add_trace(go.Pie(labels=data['Words'].head(10), values=data['Count'].head(15), textinfo='label+percent',
                             insidetextorientation='radial', marker=dict(colors=colors, line=dict(color='DarkSlateGrey')),
                             name='Pie Chart'), 1, 2)
        # customize layout
        fig.update_layout(shapes=[dict(type="line", xref="paper", yref="paper", x0=0.5, y0=0, x1=0.5, y1=1.0,
                                       line_color='DarkSlateGrey', line_width=1)])

        # customize plot title
        # fig.update_layout(showlegend=False,
        #                   title=dict(text=" <i>10 Most Common Words</i>",
        #                              font=dict(size=18, )))

        # Customize backgroound, margins, axis, title
        fig.update_layout(yaxis=dict(showgrid=False,
                                     showline=False,
                                     showticklabels=True,
                                     domain=[0, 1],
                                     categoryorder='total ascending',
                                     title=dict(text='Common Words', font_size=14)),
                          xaxis=dict(zeroline=False,
                                     showline=False,
                                     showticklabels=True,
                                     showgrid=True,
                                     domain=[0, 0.42],
                                     title=dict(text='Word Count', font_size=14)),
                          margin=dict(l=100, r=20, t=70, b=70),
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')

        # Specify X and Y values for Annotations
        x = data['Count'].head(10).to_list()
        y = data['Words'].head(10).to_list()

        # Show annotations on plot
        annotations = [dict(xref='x1', yref='y1', x=xa + data['Count'][0], y=ya, text=str(xa), showarrow=False) for xa, ya in zip(x, y)]

        fig.update_layout(annotations=annotations)
        # fig.show(renderer='png')
        chart_fig = plotly.offline.plot(fig, auto_open=False, output_type="div")
        # print(test)
        tweets_long_string = test['tweet'].tolist()
        tweets_long_string = " ".join(tweets_long_string)

        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            if(bar_chart['Sentiment'][0] == 'Positive'):
                color = "hsl(114, 100%%, %d%%)" % random.randint(50, 70)
            elif(bar_chart['Sentiment'][0] == 'Negative'):
                color = "hsl(10, 100%%, %d%%)" % random.randint(50, 70)
            else:
                color = "hsl(210, 100%%, %d%%)" % random.randint(50, 70)
            return color


        image = np.array(Image.open('/Users/girishbudhrani/Desktop/G drive/Projects/Twiter/tp/twitter_webapp/image/twitter_image.png'))
        twitter_wc = WordCloud(background_color='white', max_words=1500, mask=image)

        twitter_wc.generate(tweets_long_string)

        # display the word cloud
        fig = plt.figure()
        # fig.set_figwidth(20)  # set width
        # fig.set_figheight(10)  # set height

        plt.imshow(twitter_wc.recolor(color_func=color_func, random_state=3),
                   interpolation="bilinear")
        plt.axis('off')
        pngImage = io.BytesIO()
        FigureCanvas(plt.gcf()).print_png(pngImage)
        pngImageB64String = "data:image/png;base64,"
        pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
        plt.clf()
        # word_fig = plotly.offline.plot(twitter_wc.recolor(color_func=blue_color_func, random_state=3),interpolation="bilinear", auto_open=False, output_type="div")

        context = {'keyword': keyword, 'count': count, 'test':data_final, 'barplot_senti': barplot_senti, 'chart_f': chart_fig,'world_cloud': pngImageB64String}
    else:
        context = {'keyword':'errro'}
    return render(request, 'index.html', context)