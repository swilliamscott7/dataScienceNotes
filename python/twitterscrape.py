#!/usr/bin/env python
# coding: utf-8

import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup as bs
import time


#to open driver
def init_driver():
    driver = webdriver.Chrome()
    #pretending to be a human interacting with a website, e.g. account for content loading with page before interacting with it
    driver.wait = WebDriverWait(driver, 5)
    return driver

#to close driver so it doesn't time out. Dont interacting with driver
def close_driver(driver):
    driver.close()
    return


#make it log into twitter
def login_twitter(driver, username, password):
    
    driver.get('https://twitter.com/')
    driver.maximize_window()
    
    username_field = driver.find_element_by_class_name('js-signin-email') #class of username input field
    password_field = driver.find_element_by_css_selector('input[type=password]') #told driver which field for password
    
    #can tell it to pretend that their is a human typing on the keyboard
    username_field.send_keys(username)
    #slow down a bit
    driver.implicitly_wait(5) #1 second
    
    password_field.send_keys(password)
    driver.implicitly_wait(5) #1 second
    
    #click login button
    driver.find_element_by_class_name('EdgeButton--medium').click()
    
    return

class wait_for_more_than_n_elements_to_be_present(object):
    def __init__(self, locator, count):
        self.locator = locator
        self.count = count
        
    def __call__(self, driver):
        try:
            elements = EC._find_elements(driver, self.locator)
            return len(elements) > self.count
        except StaleElementReferenceException:
                return False

def search_twitter(driver, query, limit):
    
    #make sure you don't intereact with searchbox before it loads!
    box = driver.wait.until(EC.presence_of_element_located((By.NAME, 'q')))
    
    #make sure box is active and no text in it when we start interacting with it
    driver.find_element_by_name('q').clear()
    
    #send our query to it
    box.send_keys(query)
    box.submit() #acts as if you've pressed enter on your keyboard after search query
    
    wait = WebDriverWait(driver, 10) #wait 10 seconds to be safe, can be tweaked later
    tweets = []
    try:
        #wait for first tweet to load
        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'li[data-item-id]')))
        #want to scroll down to last tweet until no more tweets to be loaded. 
        #while loop. While statement is true, keep scrolling
        #while True:
        while len(tweets) < limit:
            #look for all elements that have this specific selector
            tweets = driver.find_elements_by_css_selector('li[data-item-id]')
            #create tweet counter
            number_of_tweets = len(tweets)
            #make sure scrolls to last tweet every time so that it keeps loading more tweets
            driver.execute_script('arguments[0].scrollIntoView();', tweets[-1]) 
            #when it does that, then need to wait until more tweets loaded
            try:
                wait.until(wait_for_more_than_n_elements_to_be_present((By.CSS_SELECTOR, 'li[data-item-id]'), number_of_tweets))
            #if timesout then stop looping. Loop until no more elements to load
            except TimeoutException:
                break 
        
        #once everything has loaded want to extract page source
        page_source = driver.page_source
     
    #if no search results at all, page source none
    except TimeoutException:
        
        page_source=None
    
    #extraxt actual tweets
    return page_source


#now we want it to combine with login and then search and scroll down the page

driver = init_driver()

username = 'AmyGrah44942927'
password = 'Kubrick1'
query = 'sky'

login_twitter(driver, username, password)
page_source = search_twitter(driver, query, 100)

# time.sleep(10)  # commenting out as doing something in the middle (not just for show)
close_driver(driver)

soup = bs(page_source, 'lxml')

tweetss = soup.find_all('li', class_='js-stream-item')
doggotweets = []

for t in tweetss:
    try:
        author = t.find('span', class_='username').get_text()
        date = t.find('span', class_='js-short-timestamp').get_text()
        text = t.find('p', class_='tweet-text').get_text()
        tweetdict = {}
        tweetdict['author'] = author
        tweetdict['date'] = date
        tweetdict['text'] = text
        doggotweets.append(tweetdict)
    except:
        pass
