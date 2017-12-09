import tweepy

consumer_key = 'wb22Y8wyJ5dQZIqcuQQs3TOMr'
consumer_secret = 'sefxdoiTHWCrEBwSWE9zCPSb4VqTIwXe95fCTBk6qi2Y34yor3'
access_token = '836485825353277440-K6JwZiLRgwtwYlWUIwnykoszckQ08Ps'
access_token_secret = 'WFOFTLkaMPLPQoc160RhBKhleI7YmAFXQ0RJzEcOUSP5j'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


results = api.search(q="#sad", count='10000')

for result in results:
    if "RT @" not in result.text and "https:" not in result.text and "http:" not in result.text and "\\u" not in result.text:
        print(result.text)
        saving_file = open('twythonDataHSad.csv', 'a')
        saving_file.write(result.text)
        saving_file.write('\n')
        saving_file.close()
