import socket
import sys
import requests
import requests_oauthlib
import json
import bleach
from bs4 import BeautifulSoup


# Include your Twitter account details
ACCESS_TOKEN = '3350695240-bRqVTgledOZiwZFg2VJ2vcsqfL3aGHi59mR3ZJN'
ACCESS_SECRET = 'q8wIk9Hd5ddZBN8X216Xxb1AHideHO90J9zyO6UNafdBf'
CONSUMER_KEY = 'pKrouLdp4uDP20dmQyUKdSpFw'
CONSUMER_SECRET = 'nyQUf3mBwrgaHEamguhtb9PCApJmBvOV5S6zpfu2rbpiNCeKHk'
my_auth = requests_oauthlib.OAuth1(CONSUMER_KEY, CONSUMER_SECRET,ACCESS_TOKEN, ACCESS_SECRET)


def get_tweets():
    url = 'https://stream.twitter.com/1.1/statuses/filter.json'	
    query_data = [('language', 'en'), ('locations', '-130,-20,100,50'),('track','iphone')]
    query_url = url + '?' + '&'.join([str(t[0]) + '=' + str(t[1]) for t in query_data])
    response = requests.get(query_url, auth=my_auth, stream=True)
    print(query_url, response)
    return response




def send_tweets_to_spark(http_resp, tcp_connection):
	for line in http_resp.iter_lines():
		try:
			full_tweet = json.loads(line)
			tweet_text = full_tweet['text']
			tcp_connection.send((tweet_text + '\n').encode())
		except Exception as e:
			print('Mensaje', e)



TCP_IP = 'localhost'
TCP_PORT = 5556
conn = None
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)

print("Waiting for TCP connection...")
conn, addr = s.accept()

print("Connected... Starting getting tweets.")
resp = get_tweets()
send_tweets_to_spark(resp, conn)