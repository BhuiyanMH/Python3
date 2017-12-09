from twython import Twython

TWITTER_APP_KEY = 'wb22Y8wyJ5dQZIqcuQQs3TOMr' #supply the appropriate value
TWITTER_APP_KEY_SECRET = 'sefxdoiTHWCrEBwSWE9zCPSb4VqTIwXe95fCTBk6qi2Y34yor3'
TWITTER_ACCESS_TOKEN = '836485825353277440-K6JwZiLRgwtwYlWUIwnykoszckQ08Ps'
TWITTER_ACCESS_TOKEN_SECRET = 'WFOFTLkaMPLPQoc160RhBKhleI7YmAFXQ0RJzEcOUSP5j'

t = Twython(app_key=TWITTER_APP_KEY,
            app_secret=TWITTER_APP_KEY_SECRET,
            oauth_token=TWITTER_ACCESS_TOKEN,
            oauth_token_secret=TWITTER_ACCESS_TOKEN_SECRET)

search = t.search(q='#angry', count=1000, lang='en')

tweets = search['statuses']

for tweet in tweets:
    if "RT @" not in tweet['text'] and "https:" not in tweet['text'] and "http:" not in tweet['text'] and "\\u" not in tweet['text']:
        print (tweet['text'],'\n')