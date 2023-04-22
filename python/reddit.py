# PRAW objects: 

# - reddit instance
# - subreddit instance 
# - submission instances
# - redittor instance (author)
# - comment/commentForest instances (submissions have a comment attribute that produces a commentForest instance)
#     - iterable 
#     - there will periodically be 'MoreComments' instances scattered throughout the forest
#         - replace these using replace_more() - must be done after 'comment_sort' is updated
#     - represents the top-level comments of the submission by the default comment sort (best)
#         - if want sorted differently, change the submission instance first using attribute 'comment_sort'
#         - submission.comment_sort = 'new'
#     - Can instead iterate over all comments as a flattened list using list(comment_instance) 
   
  
!pip install praw 
!pip install spacy

import praw
import nltk
import pandas as pd
import datetime as dt 
import pprint

# ## Using hyperparameters, look to reduce the scope of the web scraping activity: 

# 1. Name of show = Seven Worlds, One Planet
# 2. Comments_from_date = (will be release date of show) 
# 3. Comments_to_date  = (maybe one month on) 


# create an instance of Reddit
reddit = praw.Reddit(client_id='', 
                     client_secret='',
                     user_agent='viewing_sentiment')
subreddit = reddit.subreddit('all')
# then need to search on this using 'Watchmen' or some kind of search term 
watchmen_reddit_dict = {'title' = [],
                        'id' = [],
                        'created' = [],
                        'comms' = []}
foo = [] 

for submission in reddit.subreddit('television').new(limit=100):
    if 'Watchmen' in submission.title:
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            foo.append(comment.body)
watchmen_reddit_dict = {'title': [],
                        'id': [],
                        'created': [],
                        'comms': []}

for submission in reddit.subreddit('television').new(limit=100):  # returns type 'ListingGenerator' which is to be iterated through 
    if 'Watchmen' in submission.title:
        watchmen_reddit_dict['title'].append(submission.title)
        watchmen_reddit_dict['id'].append(submission.id)
        watchmen_reddit_dict['created'].append(submission.created)
        submission.comments.replace_more(limit=0)
        foo = " "
        for comment in submission.comments.list():
            foo += '__'+ comment.body
        watchmen_reddit_dict['comms'].append(foo)
        
# CAN USE THE ID AS AN EXTENSION TO FOLLOWING URL: 
# https://www.reddit.com/r/television/comments/eaci5n
# watchmen_reddit_dict
df = pd.DataFrame(watchmen_reddit_dict)
df
# topics_data
comms = [] 
url = "https://www.reddit.com/r/television/comments/"
list_ids = list(topics_data['id'])
for id in list_ids:
    submission_url = ''.join(url+id)
    #print(submission_url)
    submission = reddit.submission(url=submission_url)
    submission.comments.replace_more(limit=0)
    for comment in submission.comments.list():
        comms += comment.body
submission = reddit.submission(id='39zje0')
subreddit = reddit.subreddit('NetflixBestOf') # Needs to be a subreddit page
subreddit.description
# to recode 'created' date 
def get_date(created):
    return dt.datetime.fromtimestamp(created)
_timestamp = topics_data["created"].apply(get_date)
topics_data = topics_data.assign(timestamp = _timestamp)

topics_data.to_csv(r'C:\Users\SSC24\OneDrive - Sky\Research Analytics\Social Media Analytics\reddit_data_NetflixBestOf.csv', index=False) 
this will likely lead to this exception error ('MoreComments object has no attribute body')- i.e. submission’s comment
forest contains a number of MoreComments objects. These objects represent the “load more comments”, and “continue 
this thread” links encountered on the website.   
top_subreddit = subreddit.top(limit=2)
submission.comments.replace_more(limit=0)    # replaces or removes 'MoreComments' object from comment forest - limit of 0 removes all
# get all comment ids for all submissions 
comments = submission.comments.list()     # turns it from type 'commentforest' to type 'list'

# gets all comments for all submissions 
for top_level_comment in comments:
     print(top_level_comment.body)

# N.B. risk that item in comments list could be 'MoreComments '

# To iterate over all the replies: 
for top_level_comment in submission.comments:
    for second_level_comment in top_level_comment.replies:
        print('reply: '+second_level_comment.body)
submission.comments.replace_more(limit=None)
for comment in submission.comments.list():
    print(comment.body)
# Get comments from a submission by creating a Submission object and looping through the comments attribute
# Subreddit or specify a specific submission using reddit.submission and passing it the submission url or id.
# N.B each submission can referred to by unique id or URL. The unique id can be found in url e.g.
# id="e0r713"   for https://www.reddit.com/r/NetflixBestOf/comments/e0r713/us_end_of_watch_2012_shot_documentarystyle_this/   
submission = reddit.submission(url="https://www.reddit.com/r/NetflixBestOf/comments/e0r713/us_end_of_watch_2012_shot_documentarystyle_this/")
# submission = reddit.submission(id="a3p0uq")

submission.comments.replace_more(limit=0)
for comment in submission.comments.list():
    print(comment.body)
# To indefinitely iterate over new submissions to a subreddit use: 
subreddit = reddit.subreddit('all')   # can join multiple using + # if want all subreddits use 'all' 
for submission in subreddit.stream.submissions():
    # do something with submission
    
# filter these new submissions using their titles: 
questions = ['what is', 'who is', 'what are']
normalized_title = submission.title.lower()
for question_phrase in questions:
    if question_phrase in normalized_title:# %% [markdown]
# # Natural Language Processing 
# 
# - nltk = natural language tool kit

# %% [markdown]
# ### Word Tokenisation 
# - Splitting a sentence into its constituent words. However, the split method used above would not capture individual tokens such as "Sky News" or "Sky Broadband"
# 

# %%
# nltk.download('punkt')
from nltk import word_tokenize

text_snippet = watchmen_df.iloc[1,3]
tokens = nltk.word_tokenize(text_snippet)

# %% [markdown]
# ### PoS (Part of Speech) tagging - extracts grammatical structures in languages 
# 
# - Only want to extract the significant tokens like adjectives/adverbs

# %%
# nltk.download('all')
token_list = nltk.pos_tag(tokens) # 'JJ' adjectives
adj_list = [i for i, v in token_list if v == 'JJ']   # 'PRP','VBP', 'VBG', 'NNP'
# bag-of-words
watchmen_dictionary = {x:adj_list.count(x) for x in adj_list} 
# watchmen_dictionary

# %% [markdown]
# ### Reduce noise by removing stopwords 

# %%
# nltk.download('stopwords')
from nltk.corpus import stopwords

# Display English stopwords
stop_words = stopwords.words('English')
print(stop_words)     # print([i.upper() for i in stop_words])

# %%
sentence = 'The dog jumped over the fence'
sentence_words = nltk.word_tokenize(sentence)
sentence_words
sentence_stop_words = ' '.join([word for word in sentence_words if word not in stop_words])
sentence_stop_words

# %% [markdown]
# ### Text Normalisation: Different variations of text get converted into a standard form. For example, the words "does" and "doing" are converted to "do".
# 

# %%
sentence_2 = 'I travelled from the US to GB'
normalised_sentence = sentence_2.replace('US', 'United States').replace('GB', 'Great Britain')
print(normalised_sentence)

# %% [markdown]
# ### Stemming - extracts the underlying word stem 

# %%
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize('products')

# %% [markdown]
# # Finally apply a sentiment classifier
# 
# - pass the tokens to a sentiment classifier which classifies the tweet sentiment as positive, negative or neutral by assigning a polarity between -1 to 1 (i.e. min max scale) 

# %% [markdown]
# # Sentiment Classifier 1 - VADER (Valence Aware Dictionary and sEntiment Reasoner)
# - pip install vaderSentiment
# - VADER is a lexicon and rule-based sentiment analysis tool 
# - Is specifically attuned to sentiments expressed in social media e.g emojis, slang
# - Uses a combination of a sentiment lexicon as a list of lexical features (e.g., words) which are generally labeled according to their semantic orientation as either positive or negative
# - Four scores given - the sum of neg,neutral,positive will always equal 1
# - compound = aggregated score (sum of all lexicon ratings)- which have been normalized between -1(most extreme negative) and +1 (most extreme positive)
# 

# %%
def sentiment_analyzer_scores(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))
    
sentiment_analyzer_scores('The food is so good!') 
sentiment_analyzer_scores('The food is so good!!') # a second exclamation mark emphasises how positive it is  
sentiment_analyzer_scores('The food is SO good!!') # capital 'so' further improves score 
sentiment_analyzer_scores('The food is SO good, if you have no taste!!') # FAILS HERE 

# %%
# import SentimentIntensityAnalyzer class 
!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

# tester 

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))
    
sentiment_analyzer_scores('The food is so good!')   
sentiment_analyzer_scores('The food is so good!!')   
    

def sentiment_scores(sentence): 
  
    # Initalise a SentimentIntensityAnalyzer object. 
    sid_obj = SentimentIntensityAnalyzer() 
  
    # 'polarity_scores' method gives a sentiment dictionary which contains pos,neg,neu and compound scores 
    sentiment_dict = sid_obj.polarity_scores(sentence) 
    print("Overall sentiment dictionary is : ", sentiment_dict) 
    print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative") 
    print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral") 
    print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive") 
    print("Sentence Overall Rated As", end = " ") 
  
    # decide sentiment as positive, negative and neutral 
    if sentiment_dict['compound'] >= 0.05 : 
        print("Positive") 
  
    elif sentiment_dict['compound'] <= -0.05 : 
        print("Negative") 
  
    else : 
        print("Neutral") 

# Driver code 
if __name__ == "__main__" : 
  
    print("\n1st statement :") 
    sentence = "Geeks For Geeks is the best portal for the computer science engineering students." 
  
    # function calling 
    sentiment_scores(sentence) 
  
    print("\n2nd Statement :") 
    sentence = "study is going on as usual"
    sentiment_scores(sentence) 
  
    print("\n3rd Statement :") 
    sentence = "I am vey sad today."
    sentiment_scores(sentence) 

# %%
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

# %%
sid = SentimentIntensityAnalyzer()

for sentence in simplified_text:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print()

# %% [markdown]
# # SENTIMENT CLASSIFIER 2 - TEXTBLOB 
# - processes textual data 
# - uses a Movies Reviews dataset in which reviews have already been labelled as positive or negative.
# - Positive and negative features are extracted from each positive and negative review respectively.
# - Training data now consists of labelled positive and negative features. This data is trained on a Naive Bayes Classifier
# - Then, we use sentiment.polarity method of TextBlob class to get the polarity of tweet between -1 to 1
# - 

# %%
!pip install textblob

# %%
# import TextBlob 
from textblob import TextBlob
gfg = TextBlob("GFG is a great company and always value their employees.") 
# using TextBlob.sentiment method 
gfg = gfg.sentiment 
  
print(gfg) 

# %%
# classify polarity like so: 
if analysis.sentiment.polarity > 0:
       return 'positive'
elif analysis.sentiment.polarity == 0:
       return 'neutral'
else:
       return 'negative'
    
# Can then find the percentage of positive, negative and neutral tweets about a query.



        # do something with a matched submission
        break
   
   

