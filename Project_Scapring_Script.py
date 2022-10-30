#IMPORTS
import json
import tweepy
import configparser as cp
import datetime as dt
import time

#FETCHING DATA FROM TWITTER

def load_api():
    ''' Function that loads the twitter API after authorizing the user. '''

    config = cp.ConfigParser()
    config.read('config.ini')
    api_key = config['twitter']['api_key']
    api_key_secret = config['twitter']['api_key_secret']
    access_token = config['twitter']['access_token']
    access_token_secret =config['twitter']['access_token_secret']

    #Setup authentication for Tweepy
    auth = tweepy.OAuthHandler(api_key,api_key_secret)
    auth.set_access_token(access_token,access_token_secret)
    # load the twitter API via tweepy
    return tweepy.API(auth)

def get_search_phrases(filename):
    '''Funtion that fetches the hashtags from the files'''
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines

def scrape(words, date_since, max_tweets):
        '''Function that scrapes twitter data'''
        searched_tweets = []
        while len(searched_tweets) < max_tweets:
                remaining_tweets = max_tweets - len(searched_tweets)
                try:
                        tweets = tweepy.Cursor(api.search_tweets,
                                words, lang="en",
                                since_id=date_since,
                                tweet_mode='extended').items(remaining_tweets)
                        new_tweets = [tweet._json for tweet in tweets]     
                        print('found',len(new_tweets),'tweets for ',words)
                        if not new_tweets:
                                print('no tweets found')
                                break
                        searched_tweets.extend(new_tweets)
                except tweepy.errors.TweepyException:
                        # since there is a limit on number of tweets that can fetched via API method, we wait for 15 mins for next set of data
                        print('exception raised, waiting 15 minutes')
                        print('(until:', dt.datetime.now()+dt.timedelta(minutes=15), ')')
                        time.sleep(15*60)
                        break # stop the loop
        return searched_tweets

def write_tweets(tweets, filename):
    ''' Function that appends tweets to a file. '''
    with open(filename, 'w') as f:
        json.dump(tweets, f)

def get_tweets(type):
    # declaring variables to be used
    filename = type + 'Tweets2.json'
    date_since = "2021-04-01" # since date is used to get tweets later than this date since this was time around which vaccines started coming in for public
    search_phrases = get_search_phrases(type+'Keywords.txt') #search phrases are given in a file
    numtweet = 100 # number of tweets you want to extract in one run
    total_tweets = []
    for word in search_phrases:
        total_tweets.extend(scrape(word, date_since, numtweet)) #extract tweets for each phrase
    print('Scraping has completed!')
    write_tweets(total_tweets, filename)
    

if __name__ == "__main__":
    print('Welcome to COVID VACCINE TWEETS EXTRACTOR.')
    print('Choose which data to fetch for - AntiVaccine(A) or ProVaccine(B) or Both(C):')
    choice = input()
    print(choice)
    api = load_api()
    if choice == "A" or choice == "a":
        print('AntiVaccine Tweets will be fetch and will be stored in AntiVaccineTweets.json')
        get_tweets('AntiVaccine')
    elif choice == "B" or choice == "b":
        print('ProVaccine Tweets will be fetch and will be stored in ProVaccineTweets.json')
        get_tweets('ProVaccine')
    elif choice == "C" or choice == "c":
        print('AntiVaccine and ProVaccine Tweets will be fetch and will be stored in AntiVaccineTweets.json and ProVaccineTweets.json respectivly ')
        get_tweets('ProVaccine')
        get_tweets('AntiVaccine')
    else:
        print('Incorrect Choice. Enter A,B or C')






