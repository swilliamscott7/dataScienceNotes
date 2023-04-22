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
## Using hyperparameters, look to reduce the scope of the web scraping activity: 

1. Name of show = Seven Worlds, One Planet
2. Comments_from_date = (will be release date of show) 
3. Comments_to_date  = (maybe one month on) 
# create an instance of Reddit
reddit = praw.Reddit(client_id='WSXtg1W1qz_pug', 
                     client_secret='8XypoifpQEHKsMsXQfFyDN7S3Sc',
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
https://www.reddit.com/r/television/comments/eaci5n
watchmen_reddit_dict
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
    if question_phrase in normalized_title:
        # do something with a matched submission
        break
   
   
