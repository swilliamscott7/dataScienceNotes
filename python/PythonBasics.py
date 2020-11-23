# Python vs R # 
# Python - too many bells and whistles / package errors & dependencies / less intuitive
# R - Installing R packages is a breeze / easier to analytics 
# https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_r.html

############ install python package ###################
# New Terminal (opens up power shell) --> python -m pip install matplotlib 

import this # the zen of python 
# PEP = Python Enhancement Proposals -PEPs are python language proposals that talk about specific standards or changes that everyone should use as guidelines - PEP 8 is the de facto code style guide for Python so use when collaborating

# What is efficient code ? fast runtime (small latency between execution and returning a result) & minimal resource consumption (small mem footprint) i.e. reduced costs 


# Keyword  /Reserved words #
import keyword 
print(keyword.kwlist) 

########### Importing Modules i.e. equivalent to R's source() ######################## https://stackoverflow.com/questions/43728431/relative-imports-modulenotfounderror-no-module-named-x 
# Three variants according to directory structure etc.
from lib import dummy   # generally, should create a lib folder containing all modules (i.e. .py scripts containing functions)
import DataViz as dv    # if DataViz is in the same working directory as you're currently in, then can import directly 
from DataViz import fake_function # this means, no need for DataViz.fake_function prefix 
import directory1.subdirectory1.viz  # if nested in folders and want function from viz.py

from sklearn import * # for all packages within sklearn

dv.fake_function(8,1)   # need to indicate package name just like pd.DataFrame
dummy.fake_function(1,2)


# Some assignment python operators #
+=	x += 3	x = x + 3
*=	x *= 3	x = x * 3 
|=	x |= 3	x = x | 3
^=	x ^= 3	x = x ^ 3

##############################################################

pip freeze # in PowerShell terminal 
conda list # in PowerShell terminal 

import pandas as pd # Pandas enables R-style analytics & Modelling 
import numpy as np # 

# OS Package #
import os
os.getcwd() # get current directory - getwd() in R / pwd() in command line
os.chdir(r"C:\Users\Owner\OneDrive\Stuart\LearningPython\VSCode_Python")
os.listdir() # list files

# Memory Management # 
sys.getsizeof(range(1,10)) == sys.getsizeof([1,2,3,4,5,6,7,8,9]) # range objects take up less storage 
sys.getsizeof(df) / (1024**3)   # to convert the bytes into GB

# Environment functions # 
dir()
locals()
globals()
del(name_of_variable)

# akin to rm(list=ls()) # 
for name in dir():
 if not name.startswith('_'):
      del globals()[name]
del(name)

################ Python built-ins ####################
# i.e. python standard library - built-in types include lists,tuples,set, dict etc
# built-in functions (print(), sum(), count(), enumerate(), map(), zip() ....)
# Are optimised so should default to these, rather than building own functions 
#  
#################### Line continuation ########################
# Python joins consecutive lines if the last character of the line is a backslash - but avoid as is fragile (e.g. code breaks if unknowing space after this )
# Recommends parentheses instead !
cg_df = cg_df.groupby('channel_stacked')\
             .apply(lambda d: d.sort_values('effective_from').tail(1))\
             .reset_index(drop=True)\
             .loc[:, ['channel_stacked', 'channel_genre']]

my_very_big_string = (
    "For a long time I used to go to bed early. Sometimes, "
    "when I had put out my candle, my eyes would close so quickly "
    "that I had not even time to say “I’m going to sleep.”"
)

from some.deep.module.inside.a.module import (
    a_nice_function, another_nice_function, 
    yet_another_nice_function )

def long_function_name(
        var_one, var_two, var_three,
        var_four):
    print(var_one)
    

# Datasets to play around with #
from pydataset import data
# Check out datasets
iris_df = data('iris')
import statsmodels.api as sm; dat = sm.datasets.get_rdataset("Guerry", "HistData").data
extramaritalAffairs_df = sm.datasets.fair.load_pandas().data # discrete choice - i.e. logistic dataset
from sklearn import datasets ; from sklearn.datasets import load_boston# can artifically generate a dataset or can use an existing one
X, y = load_boston(return_X_y = True); X = pd.DataFrame(X) # gives me the data for boston housing price data, but still need the feature names 
X_dict = load_boston(); Xcols = list(X_dict['feature_names']); X.columns = Xcols # this returns a dictionary of data attributes- want feature names specifically 
X['HousePrice'] = y
centers_neat = [(-10, 10), (0, -5), (10, 5)]; x_neat, _ = datasets.make_blobs(n_samples=5000, centers=centers_neat, cluster_std=2, random_state=2) # for clustering
x_messy, _ = datasets.make_classification(n_samples=5000, n_features=10, n_classes=3, n_clusters_per_class=1, class_sep=1.5, shuffle=False, random_state=301)# for classification 
datasets.make_regression() # for regression

# Methods vs Attributes vs Functions 

# Function - Needs an input (parameters) and produces an output i.e maps an input to an output
# Methods - Need parentheses e.g. df.tail(), df.info() - doesn't need parameter 
# Attributes - No parentheses  E.G. df.columns, df.shape

# None type - corresponds to false # 
if None: 
    print('This is true')
else: 
    print('This is false')

# list - can contain heterogenous elements / mutable #

list = list(iris_df.Species.unique())
list.append('Orchid') 
list.count('Orchid')
list.sort()  # alphabetically sorts -capitals first 
list.insert(2, 'Roses') # will shift all other elements to the right 
list.extend(list2) # concats lists just like + or += (append wouldnt do this, would add a list within the original list i.e. nested list)
list.index('Orchid') # returns index of element n.b. throws a ValueError if element does not exist, so use "in" to avoid this
list.remove('Orchid') # ValueError if not present
list.reverse() 
list.pop(2) # Removes this element and returns it to indicate which is omitted
" ".join(["A", "B", "C", "D"]) 
list[-4] # negative indexing 


# dicts #

d = {"CA": "Canada", "GB": "Great Britain", "IN": "India"} 
d["GB"]
d.get("AU","Sorry") # if key does not match 'AU' returns "Sorry"
d.keys() # dict_keys object of keys
d.values() 
d.items() # Return a list of (key, value) pairs from d

# can build a dict from 2 lists using 'zip': 
keys = ['name', 'age', 'food']
values = ['Monty', 42, 'spam']
dictionary = dict(zip(keys, values)); print(dictionary)

# To access a specific part of a dict e.g. just the names, use the key:
dictionary['name']   
keys = ['name', 'age', 'food']
values = [['Monty','Giles'], [42,36],['egg','spam']]
dictionary = dict(zip(keys, values))
print(dictionary)

# tuples - immutable i.e. cannot change state / faster than lists so use if just iterating through elements / help functions return multiple values # 


####### Sets - unordered & unindexed - means for loop is only way to access item of a set - contains unique members only #######
# Pro : Compared to a list, it is a highly optimised method for checking whether a specific element is contained in the set (based on a data structure known as a hash table)

thisset = {"apple", "banana", "cherry"}
thisset.add("orange")
thisset.update(["orange", "mango", "grapes"]) # adds multiple items
thisset.remove("banana") # raises error if banana does not exist 
thisset.discard("banana") # does not raise error if banana does not exist

set1 = set([1, 4, 5])
set2 = set([1,2,2,5,8,5,10])
set1.intersection(set2)
set1.difference(set2)

# Can be very useful when looking for the differences between two lists
# Here, I want to know which indexes exist in one list but not the other 
set1 = set(df.loc[pd.isna(df["col1"]), :].index)
set2 = set(df.loc[pd.isna(df["col2"]), :].index)
diff = set1.difference(set2)

####### Matrix ########################## 

####################### Dataframes - 2D list of lists ########################################

# Three Methods : Dictionary / list of lists / read in file 
# Method 1 - Dictionary
df = pd.DataFrame({'Forename': ['Ned', 'John', 'Dany'], 'Surname':['Stark', 'Snow', 'Targaeryn']}, index=range(1,4))

# Method 2 - list of lists
df2 = pd.DataFrame([[1,2,3], [4,5,6], [7,8,9]], index=[1,2,3], columns=['a','b','c'])

# Multi-index
df3 = pd.DataFrame({'a':[1,2,3],'b':[3,4,5],'c':[7,8,9]}, index=pd.MultiIndex.from_tuples([('d',1),('d',2),('e',1)], names=['n','v'])); df3

numeric_df = pd.DataFrame(np.random.rand(10,5)); numeric_df

# df.explode() - unnests a column which contains list-like values
gamers_df = pd.DataFrame({'Publisher':['Nintendo', 'Ubisoft', 'Activision'], 'Genre':[['Shooter', 'Sports', 'Racing'], ['Shooter', 'Kids'], ['Sports', 'Adventure', 'RPG']]})
gamers_df.explode('Genre')


## Exploratory ## 

pd.crosstab(mod_classpreds, y_test, rownames = ['ClassPreds'], colnames = ['Actual'])  # equivalent to R's table() 
.value_counts() # for a pandas series
pd.read_csv() # pd.read_parquet('filename.pq')  
pd.to_csv()
iris_df.shape
iris_df.head
iris_df.columns
iris_df.columns.tolist()
iris_df.head()
iris_df.info()  - index,datatype and memory info 
iris_df.describe()                                            # Summary stats for numerical columns and will give dtype
iris_df.tail()
iris_df.dtypes
iris_df['Sepal.Length'].mean()
iris_df['Sepal.Length'].std()
iris_df['Species'].value_counts(dropna = False) # includes NAs
iris_df.count()   - gives a count of non null (non NaN) values per column 
len(iris_df.categories.unique()) # no. of unique categories in my column called categories
len(list) - number of elements in a list 
len(iris_df)   - number of rows in the dataframe 
sum(), count(), min(), max(), mean(), median()< var(), std(), quantile([0.25,0.5,0.75])
(iris_df[iris_df.columns[0]] == 5.1).sum()
iris_df.sort_values(by='names', ascending=True)
iris_df.rename(columns = {'y':'year', 'a':'age'})
iris_df.sort_index()
iris_df.reset_index() # When we reset the index, the old index is added as a column, and a new sequential index is used- We can use the drop parameter to avoid the old index being added as a column:
iris_df.drop(columns = ['bbc', 'itv'] )
iris_df.columns
iris_df['Species'].nunique() # number of unique species
pd.crosstab( pd.cut(iris_df['Sepal.Length'], 3), iris_df['Species']) # crosstab / contingency table - equivalenet to descr::crosstab() in R

################## Subsetting data - Column & Row Access #########################
iris_df['Sepal.Length'] # series dtype
iris_df[['Sepal.Width', 'Sepal.Length']] # dataframe - double squared brackets if multiple columns
iris_df.iloc[1:10,[1,3]] # using indexes
iris_df.iloc[:, [1:3]]
iris_df.iloc[[1,5,10],[2,4]]
iris_df.iloc[1, np.min(np.where(iris_df.columns == 'Sepal.Width'))] # equivalent to R's which() 
iris_df[-10:] # last n rows
iris_df[[col for col in list(iris_df.columns) if 'Sepal' in col]]  # grep approach to column selection 

# Boolean/Logical Subsetting using .loc - stands for LOgical Condition
iris_df.loc[iris_df['Sepal.Length'] > 5.2] 
iris_df.loc[iris_df['Sepal.Length'] < 5.2, ['Sepal.Length', 'Sepal.Width']] 
iris_df.loc[iris_df['Sepal.Length'] < 5.2, iris_df.columns[0:2]] 
iris_df.loc[(iris_df['Species' == 'virginica') | (iris_df['Species'] == 'setosa')]
iris_df[iris_df.Species.isin(['virginica', 'setosa'])] # isin() produces boolean series - can be used to subset rows
iris_df[~ iris_df.Species.isin(['virginica', 'setosa'])] # excludes virgin, setosa i.e. tilda is same as !=

df_filtered = df.loc[(df>0.8).any(1)]   # any(1) returns any true element over the axis 1 (rows)
df = df['sepal length (cm)'].apply(pd.to_numeric)  # Change the dtype of a column

# use of query() to filter results - requires boolean param

# subset columns according to dtypes 
df_cat = df.select_dtypes(include = ['O', 'bool', 'category'])
df_num = df.select_dtypes(include = ['float64', 'int64'])
df_numeric = df._get_numeric_data()

# new column #
iris_df['sepalArea'] = iris_df['Sepal.Length'] * iris_df['Sepal.Width']
brics['name_length'] = brics['country'].apply(len) # creates new col which is the length of the char string in another column
iris_df = iris_df.assign(SepalArea = iris_df['Sepal.Length'] * iris_df['Sepal.Width']); iris_df.head()



# change dtype 
iris_df.sepalArea.astype(int) 

############## Reshaping data ###################
pd.melt(df) # columns to rows - better for analysis
df_tidy2 = pd.melt(frame=df, id_vars='Names', value_vars=['Treatment A', 'Treatment B']);

# Pivot # 
pd.pivot(columns='var', values='val')  # rows to columns - reporting
weather = pd.DataFrame (np.array([['2010-01-01', 'min', 15], ['2010-01-01', 'max', 18], ['2010-02-01', 'min', 22], ['2010-02-01', 'max', 25]])\
    ,columns = ['Date', 'Element', 'Value'])
weather_tidy = weather.pivot(index='Date', columns='Element', values='Value') # index = col to stay fixed; columns = col to pivot into new columns; values = to fill new df with


#################### Data Cleaning ###############################
deduped_df = iris_df.drop_duplicates()
# Handling Missing Values # 
iris_df.isna().sum() # missing values per column
iris_df.fillna()  
df[['a', 'b']] = df[['a','b']].fillna(value=0) 
iris_df.dropna() # drop all NaN rows
df.dropna(axis=1) # Drop all columns that contain null values
df.dropna(axis = 1,thresh = n) # Drop all rows have have less than n non null values
iris_df.loc[iris_df['Species'].isna(), 'Region'] = 'Not Applicable'

nullRows = iris_df[iris_df.isnull().any(axis=1)]    # identifies the rows where any of the variables contain a null

iris_df[iris_df.Species.notnull()]
iris_df.iloc[1,2] = np.nan # sets NA 
iris_df.iloc[1:10,2].isnull()
iris_df.iloc[1:10,2].notnull()

# Editing values #
iris_df.loc[iris_df['sepalArea'] > 30 , 'Species'] = 'Giant virgina' 

df.loc[:, 'genre'] = df.genre.replace({'Gaming': 'GamingDating'})

iris_df['Species'].cat.add_categories('Virginica', inplace = True)
iris_df.loc[iris_df['Species'].str.contains('virgin'), 'Species'] = 'Virginica'

# One hot encoding # 
foo = pd.concat([iris_df, pd.get_dummies(iris_df.Species)], axis = 1) # Preserves order - a little like R's cbind()
foo = foo.drop('Species', axis = 1) # also, need to drop one MECE column to avoid dummy variable trap

# Categorical dtype - i.e. factors - need to add level # 
foo['days_until_contract_end_binned'] = foo['days_until_contract_end_binned'].cat.add_categories(['Not Applicable'])

# Renaming columns # 
df.columns = ['a','b','c']
df.rename(columns = lambda x: x + 1) # mass renaming
df.rename(columns = {'old_name': 'new_ name'}) # Selective renaming

# Changing dtypes #
# firstly, to identify dtype of series, need : 
iris_df['Species'].dtype  # dtype('o') means python object - specifically string
iris_df['Species'] = pd.Categorical(iris_df['Species']); iris_df['Species'].dtype # creates factor  - saves a tonne of memory - Pandas will only use unique values internally 
iris_df.info() # highlights that changing it to factor type saves memory
df.col1 = pd.to_datetime(df.col1, unit='s')  # seconds 
iris_df['Sepal.Length'] = iris_df['Sepal.Length'].astype(int)
# to change element dtypes in list, use map
map(int, ['1', '2', '3'])
# or, use list comps
a = ['1', '2', '3'];  [int(s) for s in a]

# replace() 
s.replace(1,'one') #  Replace all values equal to 1 with 'one'
s.replace([1,3],['one','three']) # Replace all 1 with 'one' and 3 with 'three'

# Working with dates #
'YYYY-MM-DD HH:MM:SS' # datetime where supported range is: '1000-01-01 00:00:00' to '9999-12-31 23:59:59'
'1970-01-01 00:00:01' UTC to '2038-01-09 03:14:07' UTC  # supported range for timestamp - Universal Time coordinated - gets converted to current time zone (for that server)
timestamp_col.dt.year # Convert to datetime, before can extract .year/.month/.dayofweek

#################### JOINS / APPENDING ETC ############
pd.concat([df1, df2], axis = 1) # or axis = 'columns' # R's cbind - order preserved 
iris_df.append(iris_df) # R's rbind()
df1.join(df2, on = col1, how = 'inner')
pd.merge(adf,bdf, how='left' , on='x1')  # how = ['right', 'inner', 'outer']

concat([df1, df2]) # appends rows  --- N.B.  ignore_index=True  # resets row index 

# Sort Values
iris_df.sort_values(['Sepal.Width', 'Sepal.Length'], ascending = [False, True] )

######### Summarising data ############

df = pd.DataFrame({'Forename': ['Ned', 'John', 'Dany', 'Ned'], 'Surname':['Stark', 'Snow', 'Targaeryn', 'Stark']}, index=range(1,5))
df2 = df.groupby(['Forename', 'Surname']).size().to_frame('Count')
# Pandas Series.to_frame() function is used to convert the given series object to a dataframe.
foo = df.drop_duplicates()
foo['count'] = df.groupby(['Forename', 'Surname']).agg({'count' : np.size})
tips.groupby('day').agg({'tip': np.mean, 'day': np.size})
mean_mins = df.query('{} > {}'.format(mins_col, th)).groupby(group_cols)[mins_col].mean()

# How to get value count per rows - akin to select *, count(*) from.... group by 1,2,3,4

foo = df.pivot_table(index = ['Forename', 'Surname'], aggfunc='size')

foo = pd.read_csv('tester.csv')



# groupby # 
iris_df.groupby('Species') # Returns a groupby object for values from one column
iris_df.groupby('Species').mean() # mean across all cols
iris_df.groupby(['Species', pd.qcut(iris_df['Sepal.Length'],3)]).mean()
df.pivot_table(index=col1,values=[col2,col3],aggfunc=mean) | Create a pivot table that groups by col1 and calculates the mean of col2 and col3
iris_df.groupby('Species').agg(np.mean) 

iris_df.groupby('Species', sort = False)["Petal.Width"].mean()
iris_df.groupby(['Species', 'Col2'] sort = False)["Petal.Width"].mean()

iris_df.groupby('Species')['Sepal.Length'].mean() #### is the same as:  iris_df.groupby('Species').mean()['Sepal.Length']

iris_df.groupby('Species').size() # use size() rather than count() as count() is the number of non-nulls per column 
annual_sales = df[['year', 'car_sales']].groupby('years', as_index=False).mean()

iris_df.groupby('Species').agg(np.mean)

cg_df = cg_df.groupby('channel_stacked').apply(lambda d: d.sort_values('effective_from').tail(1)).reset_index(drop=True).loc[:, ['channel_stacked', 'channel_genre']]

# Window Functions # 
.shift(1)     - preceding value 
.shift(-1)    - lagged by one
.rank(method='dense')  - rank with no gaps
.rank(method='min')    -rank - skip one when tied and take lowest rank 
.rank(pct=True)      - ranks rescaled to interval [0,1]
.rank(method='first')   ranks - ties go to first value 
.cumsum()
.cummax()
.cummin()
.cumprod()

############## generators - like a list comprehension, except doesn't store list in memory #################
# returns generator object (iterable) - to process big data w/o allocating excessive mem simultaneously

[num for num in range (10**1000)] # session disconnected - insufficient computer memory 
(num for num in range (10**1000))
[col for col in columns if col.startswith(genres)] 

Can then do lazy evaluation i.e. next(result) to see outcome

# Creating a new list requires more work and uses more memory. If you are just going to loop through the new list, use an iterator instead (e.g. generator)
# Only use list comprehensions when you really need to create a second list i.e. using the result multiple times
# If logic is too complicated for a short list comprehension or generator expression, consider using a generator function instead of returning a list.

############# for loops #######################
# Never use a 'for loop' in pandas DF- create a function and then use .map() or .apply() on the array(s)/column(s) 
# avoid method chaining in loop e.g. col.str.upper - requires eval on each loop - If instead you assign var = col.str.upper and then use this in loop will be fasted  
# - list comprehensions to replace loops  

world = {'Afghanistan':20000, 'Albania':22000, 'Argentina':27000}
for key, value in world.items() :
    print(key + ' has a GDP per capita of $' + str(value))

# n.b. use += in for loop to add incremements - E.G. LOttrry ticket example : 

deck, simulations, coincidences = np.arange(1, 14), 10000, 0
for i in range(simulations):
    draw = np.random.choice(deck, size = 13, replace=False) # no replacement of crds 
    coincidence = (draw == list(np.arange(1, 14))).any() # any matches i.e. draw card num 5 on draw 5 etc.
    if coincidence == True:
        coincidences += 1
coincidences

############# iterators ######################## 

# An object that has an associated next() method
# Iterables include : lists / strings / dictionaries / file connections 
# applying iter() to an iterable, creates an iterator - can identify an iterator using print() or .next()
value = iter(range(10,20,2))
print(next(value)); print(next(value)); print(next(value))

word = 'Dad' # rather than 'for i in word: print(i) ' use the below: 
it = iter(word)
print(next(it)); print(next(it)); print(next(it))   # will get an error once iterated through them all 

############### The Asterisk * ###########################
# The asterisk unpacks the iterable object to give the individual elements- just place * left of the object 
foo = iter('Dad'); print(*foo)

# Can assign first element(s) of list/tuple/set to a variable and remaining to another:
nums = [i for i in range(1,6)]
a,*b = nums

# Can combine different Iterables
nums = [1, 2, 3]; nums2 = (4, 5, 6); nums3 = {7, 8, 9}
_list = [*nums, *nums2, *nums3]; _list # unpacks it into a list 
_tuple = (*nums, *nums2, *nums3); _tuple
_set = {*nums, *nums2, *nums3}; _set

indexed_names_unpack = [*enumerate(names, 0)]

# Can be used to pack various elements - of non-distinguished length
def average(*nums):
    return sum(nums) / len(nums)
print(average(1, 2, 3, 4, 5))
# E.G.2. 
def _object(name, **properties):
    print(name, properties)
_object("Car", color="Red", cost=999999, company="Ferrari")



############### Enumerate ##################################
# creates index,item pair within an enumerator object i.e. automatic counter
# can take any iterable as an argument e.g. a list - and returns a special enumerate object
# useful in for-loops to avoid the need to create and initialise a counter 

names = ['Jerry', 'Kramer', 'Elaine', 'George']
mins_late = np.array([*range(10,50,10)])
guest_lateness = [(names[i],time) for i,time in enumerate(arrival_times)] # pairs guest with the number of minutes they were late 
print(guest_lateness)

for index, word in enumerate(words): #... etc.

############### zip() ######################################
# sticks together iterables 
# returns an iterator of tuples 
# can be unzipped using an asterisk 
# useful when modelling - lr.coef_ will return coefficients for logistic regression but not which feature they correspond to so: 
dict(zip(X.columns, abs(lr.coef_[0]))))


avengers = ['hawkeye', 'ironman']
names = ['barton', 'stark']
z = zip(avengers, names)
print(type(z)); print(*z) # asterisk to unpack iterable object
z_list = list(z) 
print(z_list)
# or use for loop to unpack 
for z1,z2 in zip(avengers, names):
    print(z1,z2)

##################### lambda anonymous functions - functions on the fly #######################

grades = [{'Name':'Jane', 'Score':96}, {'Name':'Mark', 'Score': 102}, {'Name':'Sam', 'Score':98}]
max(grades, key = lambda x: x['Score'])

# Lambda Use Cases # 
# 1. Map Function -  applies the function to all elements in the sequence (iterables)
nums = [2,3,4,5]
map (lambda num: num**2, nums) # square all elements - produces map object
lambda nums: map(lambda i: i/2, filter(lambda i: not i%2, nums)) # to get half of all even numbers in a list called nums:

# 2. filter
fellowship= ['Frodo', 'Bilbo', 'Sam']
members = filter(lambda member: len(member) > 3, fellowship)
print(list(members))

# 3. reduce - concats 
from functools import reduce 
stark = ['rob','ned','sansa']
result = reduce(lambda item1,item2: item1 + item2, stark)
print(result)


#### Map #####

# In R, can simply multiply two numeric vector by each other for elementwise multiplication - cannot do this in Python, need map ORRRRRR use numpy arrays

a =[5,2,3,1,7]
b =[1,5,4,6,8]
list(map(lambda x,y: x+y,a,b)) # Element wise addition with map & lambda
list(map(lambda x,y: x-y,a,b)) # Element wise subtraction
list(map(lambda x,y: x*y,a,b)) # Element wise product
list(map(lambda x: x**2,a))) # Exponentiating the elements of a list


names = ['simon', 'mike', 'dorothy']
upper_names = map(str.capitalize, names)
[i for i in upper_names]; # or [*upper_names] to unpack 

nums = [1.4, 8.4, 9.8]
rounded_nums = map(round, nums); print(list(rounded_nums))


############### Copying Objects ############### list2 = list1; list2 will only be a reference to list1; i.e. binding exists
# Changes made in list1 will automatically also be made in list2
# Binds a target and an object so list 1 and list 2 share the same reference 
a = [[0,1],[2,3]]
b = a
b[1][1] = 100
print(a,b)     # i.e. by changing some element in b, it has a side effect on a too
print(id(a)==id(b))
# shallow copy using .copy() - different references for lists; but elements of list binded
c = b.copy(); print(id(c)==id(b))
c = b.copy(); print(id(c[0])==id(b[0]))  # elements have same reference, meaning if you amend element in c, side effect is that it changes b too
# deep copy to copy everything w/o binding 
import copy
c = copy.deepcopy(b); id(c[1]) == id(b[1])

# Using apply() - way more efficient than looping through rows # 
iris_df.apply(np.mean)
iris_df.apply(np.max,axis=0) # axis = 1 for rowwise

########################## List Comprehensions - lapply() equivalent - enables vectorisation ################################

# More efficient than 'for loop's  - Enables vectorisation + parallel processing 
# Can build list comps over all objects except integer objects - in this case, need to create a zip object which is iterable - see below 

[i**2 for i in range(1,10)]

# Conditional list comprehension # 
[ num ** 2 for num in range(10) if num % 2 == 0]
[ num ** 2 if num %2 == 0 else 0 for num in range(10) ]
channel_df = df.loc[:, [cc for cc in df.columns if cc.startswith('channel_') and cc.endswith('_mins')]]

# Dictionary Comprehensions 
{num: -num for num in range(1,10)}

# Integer lists are not iterable: 
a = [1, 4, 9, 10, 15]
b = [4, 9, 10, 34, 7]
c = zip(a,b)
[x * y for x, y in zip(a, b)]    # efficient way of doing elementwise multiplication 

# nested list comprehensions 
matrix = [[col for col in range(5)] for row in range(5)]

############### Functions #########################
def power (number, pow = 2):
    ''' Raise argument to the power of 2, or whatever power is stated '''
    result = number ** pow
    return result 
power(8)

# **kwargs = Key-Word-ARGumentS - used if there exists a dict in the def function
def report_status(**kwargs):
    '''print out star wars characters'''
    print('Begin Report')
    for key, value in kwargs.items():
        print (key + ':' + value )
    print('End Report')
report_status(name = 'Luke', affiliation = 'jedi', status = 'missing')

# Keyword only arguments after * 
def _sample(*, name):
    return(name)
_sample('Datacamp') # creates error 
_sample(name='Datacamp') # correct 

# Functions can return multiple values using tuples: 

def raise_both(value1, value2):
    '''raise value 1 to power of value 2 and vice versa'''
    new1 = value1 ** value2
    new2 = value2 ** value1
    return(new1, new2) # type tuple
raise_both(2,3)


######### Sampling ##########
iris_df.sample(frac=0.5)      # randomly selects a fraction of rows 
iris_df.sample(n=10)          # randomly select n rows 

####################### Numpy arrays - uses optimised, pre-compiled C code ###############

# N.B. Arrays will sometimes refer to lists etc. This section specifically looks at numpy arrays 
# vector operations on numpy arrays are more efficient than native Pandas Series #
# Use numpy rather than list for fast and memory efficient alternative due to homogeneous dtype --> eliminates the overhead of checking each data type 
# Allows for elementwise multiplicatin/addition etc, just like in R 
# key for matrix manipulation 
# Create an array either using numpy functions or directly from a list 
arrayoneD = np.array([4,5,6])
arraytwoD = np.array([(4,5,6), (7,8,9)])
np.arange(20) # array of numbers 
# to create a matrix either: 
matrix = np.arange(20).reshape(4,5) # create 2D array i.e. matrix
matrix2 = np.array([[1,2,4], [5,6,9]])
array.shape 
np.argsort()
np.zeros((2,4))
np.ones((3,4))
np.eye(3,3)
foo_df = pd.DataFrame(np.array(np.random.randint(1, 100, 100)).reshape(10,10))
np.exp(3)
np.var() # variance
np.std() # 
np.sqrt()
np.dot() # elementwise matrix multiplication
np.array() * np.array() # used for array style multiplication- not matrix multiplicatin 
np.percentile(df['col'], [25,50,75])   # just pass a list of the percentiles you want 
# if an array is empty, then len(array) = 0

cars_array = np.array(['Honda', 'Volvo', 'BMW'])
cars_array = np.append(cars_array, "Renault")
np.where(cars_array == 'Renault')
np.nan

a =[5,2,3,1,7]; b =[1,5,4,6,8]
a=np.array(a); b=np.array(b)
#Element wise addition
print(a+b)
#Element wise subtraction
print(a-b)
#Element wise product
print(a*b)
# Exponentiating the elements of a list
print(a**2)



list_example =  ([2,4,5], [3,2,5])
nums2_np = np.array(list_example)
# To access the first element of the first list:
list_example[0][1]      # this is more verbose than numpy accessing
nums2_np[0,1]
# Return the first column of values 
[row[0] for row in list_example]
nums2_np[:,0]
nums = nums2_np[1,:] # Returns second row
np.argmax() # extracts index of max value - useful when extracting the class with max prob in a multinomial logit
foo = np.array([2,3,10]); np.argmax(foo)
x = np.array([[1.,2.],[3.,4.]]); x.T # transposes the array

dummy = np.random.rand(5)   # will give a number between 0-1 (pseudo random number- as comes from a seed)
np.logical_and(dummy>0.3, dummy<0.4)
np.logical_or(dummy>0.5, dummy==0.1)
np.logical_not(dummy != 0.5)     # double negative 
0.3 < dummy.all() < 0.4
0.3 < dummy.any() < 0.4  
# I only want the elements where the boolean condition is met: 
print(dummy[np.logical_and(dummy>0.2, dummy < 0.5)])

############################## strings #################
%s  # placeholder for a string 
%d  # placeholder for a number (integer)
%f  # Floating point numbers
%   # percentage 
'Hi %s I have %d donuts' %('Alice', 42) # using placeholders

s = 'hello'
s.startswith('he') # can add index args
s.endswith('lo')
s.replace('e','o')
s.title() # capitalises first letter of each word
iris_df.columns.str.contains("Sepal")
phrase = 'this_is_one_sentence'; phrase.split('_')
text = 'Programming is easy'; text.startswith(('python', 'Programming'))
'-'.join(list(map(str, range(1,10))))
# translate() method & maketrans()  - n.b. maketrans() can either have 1/2/3 arguments. If 1, needs to be dict #
# if DNA_strand "ATTGC" returns "TAACG" and DNA_strand "GTAT"  returns "CATA", how to translate any future DNA strands 
import string
def DNA_strand(dna):
    return dna.translate(string.maketrans("ATCG","TAGC"))   # maketrans provides the mapping and is usable with the translate() method - here index 0 mapped to index 0, 1 to 1 etc.
# example2
dict = {"a": "123", "b": "456", "c": "789"}
string = "abc"
print(string.maketrans(dict))

########### Control Flow Statements ###########################
z = 5
if z % 2 == 0: 
    print('z is even')
else:
    print('z is odd')


# while loop 
number = 20
while number > 4 : 
    number = number / 4 
    print (number)

# Sampling - random distributions - Pseudo random i.e. deterministic, given the np.random.seed()  
np.random.seed(123)
np.random.rand(2)
np.random.randint(0,2)  
np.random.randint(100, size=(2,8))
np.random.normal(mu, sigma, size)
# random shuffling
foo = ['dog', 'cat', 'hare']; np.random.shuffle(foo); foo

############# Distributions ################################


from numpy.random import randint
means = [randint(1, 7, 30).mean() for i in range(1000)] # list of 1000 sample means of size 30
# bernoulli - coin flip 
from scipy.stats import bernoulli
data = bernoulli.rvs(p=0.5, size=1000)
plt.hist(data)
plt.show()
####### BINOMIAL #########  - 10 basketball shots with prob of 80% of getting it in: 
from scipy.stats import binom
data = binom.rvs(n=10, p=0.8, size=1000)
plt.hist(data)
plt.show()
###### NORMAL ######### 
from scipy.stats import norm
data = norm.rvs(size=1000)
plt.hist(data)
plt.show()




################# File Handling - Opening Files ############################

# open(), close(), read(), write() and append(). 

# Bad practice is the below as need to ensure it's closed manually 
f = open('file.txt')
a = f.read()
print a
f.close()

# Instead use with() ................This will automatically close files for you even if an exception is raised inside the with block.
#### Context Manager - opens a connection to a file - ensures resources are efficiently allocated when opening a connection to a file - creates generator object #########

with open('world_dev.csv') as file: # mode = 'r' 'w'  
    # skip the column names
    file.readline() # reads one line N.B. if byte count, may only read a part of a line
    counts_dict{}
    for j in range(0,1000) # reads the first 1000 rows 
        #split the current line into a list according to comma delimiter 
        line = file.readline(), split(',')
        #get value for first column 
        first_col = line[0]
        if first_col in counts_dict.keys():
            counts_dict[first_col]+=1
        else: 
            counts_dict[first_col]=1

with open('file.txt') as f:
    for line in f:
        print line

# Process very large datasets/files/APIs in chunks, else might put a lof of strain on the RAM 

# Dan's Example 
fs = gcsfs.GCSFileSystem('sky-uk-ids-analytics-prod')
folder = 'ids-data-science/viewing_data'
fname_stub = 'cs_PropensityMartViewingOnly1pct_aggrAccMonth'
df = None
count = 0
fns = [fn for fn in fs.ls(folder) if fn.startswith(folder + '/' + fname_stub)]
for fn in tqdm(fns):
    with fs.open(fn, 'rb') as f: # rb mode = read in binary
        df = pd.concat([df, pd.read_csv(f, index_col = None, low_memory = False)])
    count += 1


result = []
for chunk in pd.read_csv('data.csv', chunksize=1000):
    result.append(sum(chunk['x']))
    ......
    ......
    total=sum(result)
df_reader = pd.read_csv('dataset.csv', chunksize=100) # initialises a reader object 
df_subset = print(next(df_reader))

filename = 'Harry_Potter.txt'
file = open(filename, mode='r')       # mode can either be r = read,  w=write 
text = file.read()
file.close()                    # always need to do this at the end 
print(file.closed) #  checks if file definitely closed.

## Excel Workbooks ## 
file = 'urbanpop.xlsx'
data = pd.ExcelFile(file)
print(data.sheet_names)    # this gives you the name of all the sheets in excel- to then be able to decide which one you want e.g. ['Inventory', 'Sales', 'Employees']
df1 = data.parse('Inventory')
df2 = data.parse('Sales', skiprows[0], names=['Country', 'Quantity'])

############# Scope #######################
# Not all objects are accessible everywhere in a script
# Global Scope - defined in the main body of the script
# Local Scope - defined within a function 
# Built-in Scope - named in the pre-defined built-ins module python 3 provides e.g. print(), sum()
# LEGB rule: idea that Python will look in Local scope, Enclosing function (if exists), Global scope, then Built-in scope 

value = 10 
def square(val):
    '''squares the inserted value'''
    global value 
    value = value ** 2
    return value 
square(3) 


################# Error Handling #############################
def sqrt(x): 
    '''finds the square route of the numerical input'''
    try:
        return x ** 0.5
    except:
        print('x must be an int or float')
    return sqrt
print(sqrt(4))
print(sqrt('hi'))

def sqrt(x):
    '''returns sqrt of a non-negative integer'''
    try:
        if x < 0 : 
            raise ValueError('x must be non-negative')
        else: 
            return x ** 0.5
    except TypeError:
        print('X must be an int or float')
print(sqrt(-2))
print(sqrt('hi'))

####################### MISC MASH ##########################

# .format()
pubs = ['spoons', 'swan', 'jimmys']  
drinks = ['beer', 'shandy', 'wine']
for p in pubs: 
    for d in drinks:
        print('I love drinking {} at {}'.format(d,p))

# ord() gets ASCII code for character e.g. 'a' is 97 #
ord('$') # returns integer that represents the Unicode code point for the given Unicode character

# Progress bar # 
from tqdm import tqdm

# Cartesian Product - i.e. unique combinations # 
from itertools import product
[i for i in product(range(1,10),range(10,20))]
[i for i in product('ABCD', repeat=2)]
product('A', 'B', 'C')

import statsmodels.api as sm
from statsmodels.formula.api import ols
moore = sm.datasets.get_rdataset("Moore", "carData", cache = True) # load
data = moore.data
data = data.rename(columns={"partner.status" : "partner_status"}) # make name pythonic
# n.b. C() tells model to treat variable as categorical variable explictly rather than say string or integer e.g. res = ols(formula='Lottery ~ Literacy + Wealth + C(Region)', data=df).fit()
# Interaction term used- will automatically include main effect terms too ie.  C(fcategory, Sum) + C(partner_status, Sum) + C(fcategory, Sum)*C(partner_status, Sum)
moore_lm = ols('conformity ~ C(fcategory, Sum)*C(partner_status, Sum)', data=data).fit() # do not know what the Sum represents - potentially indicates to use Sum_sq ?? 
table = sm.stats.anova_lm(moore_lm, typ=2) # Type 2 Anova DataFrame
print(table)


# Cool font 
from pyfiglet import Figlet
f = Figlet(font='slant')
print(f.renderText('Random Cool Stuff with Python'))