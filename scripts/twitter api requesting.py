#!/usr/bin/env python
# coding: utf-8

import sys
import tweepy, io, json
from time import sleep,time
from datetime import datetime
import gzip
from tweepy.streaming import StreamListener

CONSUMER_KEY = 'your key'
CONSUMER_SECRET = 'your key'
ACCESS_TOKEN = 'your key'
ACCESS_TOKEN_SECRET = 'your key'

# lower left corner, can be obtained from http://www.latlong.net
MINLAT=18.96
MINLON=-8.67
# upper right corner
MAXLAT=37.09
MAXLON=12

BATCH_SIZE = 1000
PROJECT = '../../tweets'
log = open(PROJECT+'.log', 'a')

def write_tweets(tweets,empty=True):
  stop = False
  batch_no=0
  while len(tweets)>=BATCH_SIZE or (empty and len(tweets)>0):
    batch_no+=1
    batch=[e._json for e in tweets[:BATCH_SIZE]]
    output=gzip.open(PROJECT+'/'+datetime.now().isoformat()[:10]+'_'+str(batch[0]['id'])+'.gz','w').write(json.dumps(batch, sort_keys=True, indent=2, ensure_ascii=False, encoding='utf-8').encode("utf-8"))
    tweets=tweets[BATCH_SIZE:]
  if batch_no>0:
    log.write(datetime.now().isoformat()+'\tNumber of batches of tweets written: '+str(batch_no)+'\n')
  if stop:
    log.write(datetime.now().isoformat()+'\tStopping with '+str(no_tweets)+' tweets collected.\n')
    serialize()
    sys.exit(0)
  return tweets

def serialize():
  if MODE=='LANG':
    pickle.dump(user_index,open(PROJECT + '.user_index','w'))
    log.write(datetime.now().isoformat()+'\tSerialized the user index\n')


def search(term):
  try:
    result=api.search(term)
  except Exception as e:
    log.write(datetime.now().isoformat()+'\t'+str(e)+'\n')
    log.flush()
    return []
  return result

# geo_mode()
try:
  auth = tweepy.OAuthHandler(CONSUMER_KEY,CONSUMER_SECRET)
  auth.set_access_token(ACCESS_TOKEN,ACCESS_TOKEN_SECRET)
except tweepy.TweepError:
  print ('Error! Failed to get access token.')

api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True,retry_count=3,retry_delay=3600)
places = api.geo_search(query="Algeria", granularity="country", tweet_mode='extended')
place_id = places[0].id


batch_no = 0
while batch_no<15000:
  tweets = api.search(q="place:%s" % place_id)
  for tweet in tweets:
    print (tweet.text.encode("utf-8") + " | " + tweet.place.name.encode("utf-8") if tweet.place else "Undefined place\n")
    write_tweets(tweets)
