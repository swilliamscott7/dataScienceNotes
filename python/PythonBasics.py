# Everything in python is an object, names are just identifiers which are bound to an object
# below : observe that both second and first can be used to refer to the same object
def first(msg):
    print(msg)
first("Hello")
second = first
second("Hello")

# Higher order functions - functions that take other functions as arguments #
def inc(x):
    return x + 1
def dec(x):
    return x - 1
# higher order function takes either dec() or inc() when invoked
def operate(func, x):
    result = func(x)
    return result
operate(inc,3)
operate(dec,3)

# A function can return another function - below is aan example of a nested function # 
def is_called():
    def is_returned():
        print("Hello")
    return is_returned
new = is_called()
new()

##### CONSTRUCTORS #######
# Used for instantiating an object - the task of the contructor is to intiialise (assign values) to the members of the class when an 
# object/instance is created E.G. __init__() method is a constructor

class Dog:
    #default constructor
    def __init__(self, breed):
        self.tail = 'Yes'
        self.breed = breed

###### CLOSURES - these are nonlocal variables in a nested function; occur when a nested function references a value in its
# enclosing scope  #####

# Closures = Process by which some data gets attached to the code # 
# The value in the enclsoing scope is remembered even when the variable goes out of scope of the function itself is removed from
# the current namespace 

# Criteria for creating a closure: 
# Nested function + nested function must refer to a value defined in the enclosing function + enclosing function returns nested function

# Nested function below highlights how non-local variable 'msg' can be accessed by printer() 
def print_msg(msg):
    # This is the outer enclosing function
    def printer():
        # This is the nested function whcih is able to access the non-local 'msg' variable of the enclosing function
        print(msg)
    printer()
print_msg("Hello")

def print_msg(msg):
    # This is the outer enclosing function
    def printer():
        # This is the nested function
        print(msg)
    return printer  # returns the nested function
another = print_msg("Hello")
another() # it's unusual that 'Hello' is still remembered here despite already being executed by the print_msg() function

# This value in the enclosing scope is 
# remembered even when the variable goes out of scope or the function itself is removed from the current namespace.
del(print_msg)
another() # still remembered 

# USE CASES FOR CLOSURES ####
# In place of global variables and to provide some form of data hiding
# Can also provide an object oriented solution 
# If a class were to have few methods (mostly when just one), often closure provides alternate, more elegant solution ;
# But when number of attributes & methods get larger, it's better to implement a class 

#EXAMPLE USE CASE
def make_multiplier_of(n):
    def multiplier(x):
        return x * n
    return multiplier
times3 = make_multiplier_of(3)
times5 = make_multiplier_of(5)
print(times3(9))
print(times5(3))
print(times5(times3(2)))

# To identify a closure, function will have __closure__ attribute 
make_multiplier_of.__closure__ # returns nothing 
times3.__closure__ # returns tuple therefore is closure function 
times3.__closure__[0].cell_contents # shows the stored closed value


#### DECORATORS #####
# Basically, a decorator takes in a function, adds some functionality and returns it
# Is called metaprogramming because apart of the program tries to modify another part of the program at compile time

# The most common Python decorators you'll run into are:
@property
@classmethod # receives the class as an implicit first argument, just like an instance method receives the instance
@staticmethod # does not receive an implicit first argument (class)-a method which is bound to the class, and not the object of the class 

# Python program to demonstrate use of class method and static method #
from datetime import date 
class Person: 
    def __init__(self, name, age): 
        self.name = name 
        self.age = age 
       
    # a class method to create a Person object by birth year. 
    @classmethod
    def fromBirthYear(cls, name, year):     # a method that is bound to the class and not the object of the class. Can access/modify class state
        return cls(name, date.today().year - year) 
       
    @staticmethod # cannot access or modify class state. It is present in class, becuase it makes sense for the method to be present in class
    def isAdult(age): 
        return age > 18
   
person1 = Person('mayank', 21) 
person2 = Person.fromBirthYear('mayank', 1996) 
   
print (person1.age) 
print (person2.age) 
print (Person.isAdult(22))
# use cases : @staticmethod vs @classmethod - use class to create factory methods (i.e. return class object) vs static to create utility functions

# Python decorators make extensive use of closures 
# Functions & Methods are called 'callable' because they can be called (will have attribute .__call__ 
# In fact anything which implements __call__() is termed callable 
# In the most basic sense, a decorator is a callable that returns a callable 

# Example of a decorator # 
def make_pretty(func):
    def inner():
        print("I got decorated")
        func()
    return inner
def ordinary():
    print("I am ordinary")
    
    
    
############################################################################
$ pip list --format=freeze > requirements.txt ## to get all packages with version type - issue with $ pip feeze > requirements.txt is that will have @ packages

# nbconvert converts your notebook document file to another static format, such as HTML, PDF, LaTex, Markdown, reStructuredText (rather than defacto JSON format)
### THIS IS INVALUABLE - ABILITY TO RUN NON-INTERACTIVELY IN COMBINATION WITH ABILITY TO RUN VARIOUS LANGUAGES USING MAGIC COMMANDS MAKES IT GREAT FOR ETL PIPELINES OR REPORTING (IF USING SOME FORM OF SCHEDULING)

# Convert  ipynb to script - commmand line # 
$ jupyter nbconvert --to script [YOUR_NOTEBOOK].ipynb   # just copy file path for notebook 
!jupyter nbconvert --to script config_template.ipynb # can even use this to do it within the notebook itself 
jupyter nbconvert --to script *.ipynb # for all ipynb
!jupyter nbconvert --to html Untitled4.ipynb


!pip list | grep trax   # finds the list of package that match this substring - would give version for trax 

isinstance(foo, pd.DataFrame) # checks data type 

# OOP # instance vs class vs static methods 


## runing a shell command in terminal 
$ sh your_shell_file.sh

# RUNNING A PYTHON SCRIPT IN COMMAND LINE #
def main():
    parser = argparse.ArgumentParser(description='PE scoring')
    parser.add_argument('--setup_file', type=str, help='Path to the setup json file. It is used to describe the type of analysis we want to perform')
    parser.add_argument('--handle_inelastic', type=int, default = 0, help='Integer value which should be either 0 or 1 to determine whether to handle inelastic customers separately in our optimisation')
    
    args = parser.parse_args() 
    
    with open(args.setup_file) as json_file:
        analysis_dicto = json.load(json_file)
    handle_inelastic = args.handle_inelastic
    if handle_inelastic not in [0,1]:
        raise ValueError('Value must be 0 or 1, where 1 indicates handling inelastic customers outside of the optimisation algorithm')
        
    pipeline_orchestration(analysis_dicto['SCORING_DATA_PATHS'],
                           analysis_dicto['BUCKET_ID'],
                           analysis_dicto['PREPRO_OPERATIONS_DICT'],
                           analysis_dicto['MAIN_KEY'],
                           analysis_dicto['ROOT_PATH'],
                           analysis_dicto['LGBM'],
                           analysis_dicto['DISCOUNT_LEVELS'],
                           analysis_dicto['THRESHOLD'],
                           analysis_dicto['THRESHOLD_INELASTIC'],
                           analysis_dicto['BATCH_SIZE'],
                           default_csv_filename='PE_UK_scoring_data_tmp',
                           output_filename='PE_UK_rebuild_scoring_output_TP_CustomerBase',
                           log_file=analysis_dicto['LOG_FILE_NAME'],
                           handle_inelastic=handle_inelastic)
    
if __name__ == "__main__":
    main()

$ python tp_run_scoring_inelastic.py --setup_file analysis_setup_tp.json --handle_inelastic 1

# To transform into PPT style with slideshows use 'nbpresent' and 'RISE'. To turn into interactive dashboard use 'jupyter_dashboards'



#!/usr/bin/env python # if at start of file, is shebang that indicates script written in python - tells compiler I believe 
#  
############################### Jupyter Notebook Tips & Tricks  ###########
- Kernel --> interrupt # this stops the current cell running

##### Tab completion #####
# Use shift + Tab - to get doc on class/method etc. e.g. pd. (now press shift+tab)  OR   pd.read_csv() (now press shift+tab)
# For traditional autocompletion on classes & Methods try these two things if not automatically working (n.b. is generally to do with packages installed - relies on Jedi)
%config Completer.use_jedi = False
$ conda install jedi==0.17.2 # for your relevant virtual environment

# to run bash commands in ipynb cell : 
%%bash 
conda activate cine_upgrade
conda list

###### Ipython Extensions ###### = an importable Python module that has a couple of special functions to load and unload it
- To load an extension use the following magic command
% load_ext myextension
% load_ext google.cloud.biquery
# Ipython Extensions which are already bundled include:
1. %autoreload # to reload modules if make changes to imported module (i.e. .py file ) w/o need to explictly reload/restart kernel 
2. %storemagic # automatically restores stored variables at startup

# magic commands - utilise the ipython kernel (these are either line-oriented or cell-oriented)
Magic_Name,	Effect
%lsmagic  Lists all commands
%env	Get, set, or list environment variables
%pdb	interactive debugger
%pylab	Load numpy and matplotlib to work interactively
%%debug	Activates debugging mode in cell
%%html	Render the cell as a block of HTML
%%latex	Render the cell as a block of latex
%%sh	%%sh script magic
%%time	Time execution of a Python statement or expression
%prun	Do a performance run
%writefile	Saves the contents of a cell to an external file
%pycat	Shows the syntax highlighted contents of an external file
%who	List all variables of a global scope
%store	Pass variables between notebooks
%load	Insert code from an external script
%run	Execute Python code

# To mix languages in your notebook with the ipython kernel w/o the need to set up extra kernels can use the magics:
$ pip install ipython-sql    - can then use magic : %load_ext sql
$ pip install cython - to use %load_ext cython
$ pip install rpy2 - to use %load_ext rpy2.ipython; %R library(ggplot2); %%R -i df 
ggplot(data=df) + geom_point(aes(x=A, y=B, color=C))



### Nathan's code - if make changes to already imported module  
# load the "autoreload" extension which is installed by default - will only need to do so once 
%load_ext autoreload
# %aimport is to be followed by all modules we wish to automatically import or not to be imported e.g. %aimport mod1, mod2   e.g.2 %aimport -mod1 # marks module not to be autoreloaded
%aimport CR_Transformations
# set the autoreload strategy to "1" - "Reload all modules imported with %aimport every time before executing the Python code typed." N.B. %autoreload 0 = disables automatic reloading
%autoreload 1  
# show the current autoreload status,
%aimport

# To check function updated when imported in
help(my_func_name)
dir(my_func_name) # often just change the header string to have a fullstop as a quick check
 
# Or as per site : https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html 
%load_ext autoreload
%autoreload 2 # Reload all modules (except those excluded by %aimport) every time before executing the Python code typed.
from foo import some_function
some_function()   # outputs 42
# Now, open foo.py & change some_function to return 43
some_function() # automatically outputs 43 now 

# 2. Example of %storemagic - https://ipython.readthedocs.io/en/stable/config/extensions/storemagic.html 
l = ['hello',10,'world']
%store l
exit # i.e close ipython session and then reopen
l # l is not defined
%store -r # refresh all variables ...or
%store -r foo df # refreshes just variables you've named e.g. foo and df in this case  

##### END OF #######


# 'and' vs '&'
# Use and if considering a single logical condition
# Use & if considering this condition for a pandas series / array i.e. bitwise approach e.g. df['Channel'] == 'sky atlantic' ) & (df['start_ts'] == primetime_slot) vs x='sky' and y = '9pm'


############## Boiler Plate code ###################
# = section of code that can be reused over and over without change (think boilerplate for legal contracts too - same for all who sign bar name)
# Used more and more in AI/ML as there are more growing frameworks and libraries
# Requirements of boilerplate code for large projects (production ready): Good Documenation / Code structure with deeper abstraction level / 
# / has CLI tool (for rapid prototyping and setup) / Scalable / Easy testing tools / Necessary API modules / Server&Client code for setup /
# Proper Navigation and Routing Structure



#### A python project should contain : #########################################
# 1. setup.py file # to make the code installable
# This file describes the metadata about your project. only three required fields: name, version, and packages.
# name = must be unique if you wish to publish to Python Package Index (PyPI)
# version = keeps track of different releases of the project
# packages = location of the Python source code within your project

# inside setup.py #
from distutils.core import setup

setup(
    name='ModelRetrain',
    version='0.1dev',
    packages=['modelretrainMain',],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.txt').read(),
)

# 2.  README.txt file - describing an overview of your project
# 3. Unit tests for your code  (for functions, classes, methods and modules)
# 4. Preferably a command line interface - i.e. can be run from shell --- generally using ArgParse library

################################################################################
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

# If running from top directory and want to run a test.py (adventure/tests/test_dungeon.py) file in a nested subdirectory, add a init.py in this test.py directory so can reference like so:
 from ..dungeon import Dungeon # can therefore put this at top of script


# What if script is in a subdirectory and want to import from a parent directory ? #
sys.path.append("C:\\Users\\SSC24\\OneDrive - Sky\\Advanced Analytics")
# sys.path.remove("C:\\Users\\SSC24\\OneDrive - Sky\\Advanced Analytics") reverse this
import Dummy as d; d.multiply(2,3)  # where Dummy file path is "C:\Users\SSC24\OneDrive - Sky\Advanced Analytics\Dummy.py"

# What does __init__.py do? 
# Used to be required to make a directory on disk a package (with modules to impot) but since Python 3.3 no longer required
# mydir/spam/__init__.py
# mydir/spam/module.py
# given  mydir is on your path, you can import the code in module.py as:
import spam.module
# or
from spam import module
# Removing  __init__.py means Python will no longer look for submodules inside that directory, so attempts to import the module will fail
# N.B.  __init__.py file is usually empty 

# Namespace vs Regular packages - the two types in Python
# Regular packages = traditional packages - implemented as a directory containing an __init__.py file. When imported, this __init__.py file is implicitly executed
# and the objects it defines are bound to names in the package’s namespace. The __init__.py file can contain the same Python code that any other module can contain
# and Python will add some additional attributes to the module when it is imported


# Some assignment python operators #
+=	x += 3	x = x + 3
*=	x *= 3	x = x * 3 
|=	x |= 3	x = x | 3
^=	x ^= 3	x = x ^ 3

# ASSIGNMENT OPERATOR AND DIVIDE SIMULTANEOUSLY 
scale_factor = 1000.0
train_df["median_house_value"] /= scale_factor 

# OPERATORS # 
x = 2 + 3
# Just a pretty way of calling an "add" function:
x.__add__(7) # x = +(2,3) - Operators are actually methods of the first input: they're the mysterious __ functions you see when you do dir(func)
# Sometimes a type may not have an operand (thing that an )
[2, 3, 4] - [5, 6]
# Other times it might 
'hello'+'friend'
[1,2,3]+[4,5]

# 
10//3 # divide then round down
# float type is not a real number - is actually a computer representation of a real number called a "floating point number". Representing  √2 or 1/3 perfectly would be impossible in a computer - so use finite amount
N=10000.0
sum([1/N]*int(N))
##############################################################


### GOOD EXAMPLE OF DOCSTRINGS IN YOUR FUNCTIONS ###

def get_corrections(word, probs, vocab, n=2, verbose = False):
    '''
    Input: 
        word: a user entered string to check for suggestions
        probs: a dictionary that maps each word to its probability in the corpus
        vocab: a set containing all the vocabulary
        n: number of possible word corrections you want returned in the dictionary
    Output: 
        n_best: a list of tuples with the most probable n corrected words and their probabilities.
    '''
    



pip freeze # in PowerShell terminal 
conda list # in PowerShell terminal 

import pandas as pd # Pandas enables R-style analytics & Modelling 
import numpy as np # 

import pprint # for console printing 
pp = pprint.PrettyPrinter(indent=4) # Instantiate a pretty printer

# to change how pandas displays dataframes - pandas.set_option(optname, val) : 
# N.B. RECENTLY FOUND THAT THIS ONLY WORKS IF SET NUMBER OF ROWS TO BE MORE THAN THERE ARE # 
pd.set_option('display.max_rows', 500) 
pd.set_option('display.max_columns', 500) # to see all columns rather than just an elipsis
pd.set_option('display.width', 1000)
pd.get_option("display.max_rows") # to see how many can be displayed at present 
# etc.
### or google use this: 
pd.options.display.max_rows = 1000 # needs to be some arbitrarily large number that is larger than the number of rows you will be displaying for it to work
pd.options.display.float_format = "{:.1f}".format

####### SHELL COMMANDS - Can be done in python though eventually ends up in a bash or batch file ####
#### OS Package - allows you to interact with operating system ####
# os module contains two submodules os.sys and os.path  - are wrappers for platform specific modules so works on UNIX/windows/MacOS etc.
import os
os.getcwd() # get current directory - getwd() in R / pwd() in command line
os.curdir()
os.chdir(r"C:\Users\SSC24\OneDrive - Sky\Advanced Analytics\Python")
os.chdir(os.path.expanduser("~")) # To home directory 
os.listdir() # list files
os.path.dirname('/home/User/Documents/file.txt') # will print out the parent directory(s) of the file
os.path.dirname('fake_file.txt')
os.path.join(os.sep, 'home', 'user', 'work')  # adds in the necessary \\ etc. to create a valid directory address
os.path.join(os.curdir, 'my_logs')
os.path.split('/usr/bin/python')
os.remove(r"C:\Users\SSC24\OneDrive - Sky\Advanced Analytics\DummyCopy.py") # removes file 
os.mkdir('subdirectory_dummy')
os.path.exists('/home/jupyter/.ipython') # boolean result


# If save this as a script & run it, will see the output in the command line - BUT within ipynb won't get the output inline - just 0,1 for success/failure - will find the output in the command line where you have started Jupyter notebook.
os.system('ls -l')

# Environment variables are accessed through os.environ
os.environ['HOME'] / os.environ['HOME'] 
os.environ['USERNAME'] # to get username
os.environ # gives a list of all environmental variables 
print(os.environ.get('KEY_THAT_MIGHT_EXIST')) # returns None if not present rather than raise a KeyError
# To change / create an environment variable : #
os.environ['PYTHONHASHSEED']= 10

# Memory Management # 
sys.getsizeof(range(1,10)) == sys.getsizeof([1,2,3,4,5,6,7,8,9]) # range objects take up less storage 
sys.getsizeof(df) / (1024**3)   # to convert the bytes into GB

# poor approx of memory
foo = 0
for i in dir():
    foo += sys.getsizeof(eval(i))
foo/(1024**3) # GB conversion

# eval() ensures it is the object and not the memory of the string itself
getattr(sys.modules[__name__], 'df') # believe this works too 
# sys #
sys.path # list of directory locations -  Python looks in several places defined here when importing a module
sys.modules

# sys.argv - is a list in Python containing the command-line arguments passed to the script
sys.argv = ['Model_report.py',  '--product', 'product', '--namespace', 'namespace', '--original_proportion', 'original_proportion','--adjusted_proportion','adjusted_proportion']
len(sys.argv) # gives number of arguments passed
sys.argv[0] # the name of the script
str(sys.argv) # the arguments 

# Environment functions # 
dir()  # n.b. when  dir() is applied to a function, will list all attributes & methods of that function
dir(str)   # shows all the methods available on strings in python
locals()
globals()
del(name_of_variable)

# akin to rm(list=ls()) # 
for name in dir():
 if not name.startswith('_'):
      del globals()[name]
del(name)


# To get source code for an imported function or understand it better  e.g. 
from utils import lookup
help(lookup) # gives the docstring for that function
lookup?? # gives the source code for that function

# setting with copy warning
df_lag = df.loc[:,lagged_features]
df_lag._is_copy # false
df_lag._is_view # false
df_lag = df[lagged_features] 
df_lag._is_copy # true - gives ref

df_lag.fillna(-10, inplace=True) # if try to use inplace after copy() will throw warning so solutions:

df_lag = df.copy()[cols] # or 
df_lag = df.loc[;,cols] 

# check if variable exists else assign value # 
try:
    trading_imp_df
except NameError:
    trading_imp_df = pd.read_csv('outputs/trading_features_02062021_100154.csv')

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

# Python I/O
print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False) # output function 
num = input('Enter a number: ') # Takes input directly from user i.e. no longer static and dependent on source code

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

# CONTAINERS : list (mutable/iterable) / tuples (immutable) 
# Your programs will be faster and more readable if you use the appropriate container type for your data's meaning. Always use a set for lists which can't in principle contain the same data twice etc. 

# in a list we use a number to look up an element whereas in a dict we use a key


# list - can contain heterogenous elements / mutable # Is python's basic container type: a container type: its purpose is to hold other objects. We can ask python whether or not a container contains a particular item
'Dog' in ['Cat', 'Dog', 'Horse']
list = list(iris_df.Species.unique())
list.append('Orchid') 
list.count('Orchid')
list.sort()  # alphabetically sorts -capitals first - N.B. Sorts in situ so do not re-assing to variable else will store NoneType and won't get back list when try to print list out 
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
# accepts immutable keys only #

# Building an empty dictionary using a dictionary comprehension
keys = train_df.columns[0:5].tolist()
corr_dict = {key: None for key in keys}  

# can build a dict from 2 lists using 'zip': 
keys = ['name', 'age', 'food']
values = ['Monty', 42, 'spam']
dictionary = dict(zip(keys, values)); print(dictionary)
'name' in dictionary # when we test for containment in a dictionary, we test on the keys

# To transform feature values according to dict ######
dtv_tenure_transform_dict = {'<3mths':0, '3-6mths':1, '6-12mths':2, '1-2yrs':3, '3-4yrs':4, '5-6yrs':5, '7-10yrs':6, '>10yrs':7, 'Other':np.nan}
df['dtv_tenure_transformed'] = df['dtv_tenure'].replace(dtv_tenure_transform_dict)
######################################################

# To access a specific part of a dict e.g. just the names, use the key:
dictionary['name']   
keys = ['name', 'age', 'food']
values = [['Monty','Giles'], [42,36],['egg','spam']]
dictionary = dict(zip(keys, values))
print(dictionary)
page_type_dict[key].append(something)

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

# or can create a set using curly brackets without the colons:
primes_below_ten = { 2, 3, 5, 7}

x = set("Hello")
y = set("Goodbye")
x & y # Intersection
x | y # Union
y - x # y intersection with complement of x: letters in Goodbye but not in Hello

# Can be very useful when looking for the differences between two lists
# Here, I want to know which indexes exist in one list but not the other 
set1 = set(df.loc[pd.isna(df["col1"]), :].index)
set2 = set(df.loc[pd.isna(df["col2"]), :].index)
diff = set1.difference(set2)

####### Matrix ########################## 

####################### Dataframes - 2D list of lists ########################################

# Three Methods : Dictionary / list of lists / read in file 

# Create empty dataframe of specific dimension #
ohe_genres = pd.DataFrame(np.nan, index=np.arange(0,len(meta_df)), columns=unique_genres)

# Method 1 - Dictionary where values are a list
df = pd.DataFrame({'Forename': ['Ned', 'John', 'Dany'], 'Surname':['Stark', 'Snow', 'Targaeryn']}, index=range(1,4))

# Method 1a - Dictionary where value is a scalar
df = pd.DataFrame(removal_effects_dict.items(),columns=['channel_name','removal_effect_manual'])

# Method 2 - list of lists
df2 = pd.DataFrame([[1,2,3], [4,5,6], [7,8,9]], index=[1,2,3], columns=['a','b','c'])

# Method 3 : Arrays

pd.DataFrame(columns=['Show', 'Clu'], data = np.array([df.index,cluster_labels]).T)

# Multi-index
df3 = pd.DataFrame({'a':[1,2,3],'b':[3,4,5],'c':[7,8,9]}, index=pd.MultiIndex.from_tuples([('d',1),('d',2),('e',1)], names=['n','v'])); df3

numeric_df = pd.DataFrame(np.random.rand(10,5)); numeric_df

# df.explode() - unnests a column which contains list-like values
gamers_df = pd.DataFrame({'Publisher':['Nintendo', 'Ubisoft', 'Activision'], 'Genre':[['Shooter', 'Sports', 'Racing'], ['Shooter', 'Kids'], ['Sports', 'Adventure', 'RPG']]})
gamers_df.explode('Genre')

# Numpy representation of a dataframe #
# NOTE: Note that this can be an expensive operation when your DataFrame has columns with different data types, which comes down to a fundamental difference between pandas and NumPy: 
# NumPy arrays have one dtype for the entire array, while pandas DataFrames have one dtype per column. When you call DataFrame.to_numpy(),
# pandas will find the NumPy dtype that can hold all of the dtypes in the DataFrame. This may end up being dtype=object, which requires casting every value to a Python object. 
df.values # returns a numpy representation of the dataframe - i.e. axis labels removed etc N.B. Pandas recommends using to_numpy()
df.to_numpy() 

# Check if two dataframes identical 
df.equals(df2)

## Exploratory ## 

pd.crosstab(mod_classpreds, y_test, rownames = ['ClassPreds'], colnames = ['Actual'])  # equivalent to R's table() 
pd.crosstab(df['Embarked'], df['Pclass'], normalize = True) # as percentage of total 
pd.crosstab(df['Embarked'], df['Pclass'], normalize = 'columns') # as percentage of column totals 
pd.crosstab(df['Embarked'], df['Pclass'], values = df['Fare'], aggfunc = np.sum, margins = True, normalize = True) # to get row and column sums


.value_counts() # for a pandas series
pd.read_csv() # pd.read_parquet('filename.pq')  
pd.to_csv()
iris_df.shape
iris_df.head
df.sample(n=5) # better way to scope out data
iris_df.columns
iris_df.columns.tolist()
iris_df.head()
iris_df.info()  - index,datatype and memory info 
iris_df.describe()  & df.describe(include="all")                                          # Summary stats for numerical columns and will give dtype
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
iris_df.index.name = 'new_index_name' # f want to rename your index column

df.col1.compare(df.col2) # highlights the differences between different series


# To remove columns
iris_df.drop(columns = ['bbc', 'itv'] )
df_train[df_train.columns.difference(['Id'])] # this also works - will include the complements of 'Id' i.e. essentially drops 'Id' column
iris_df.columns
# PREFERABLY - to get columns as a list:
list(iris_df) # or list(iris_df.columns.values)
iris_df['Species'].nunique() # number of unique species
pd.crosstab( pd.cut(iris_df['Sepal.Length'], 3), iris_df['Species']) # crosstab / contingency table - equivalenet to descr::crosstab() in R

# Row count & Column count #
len(df) / df.shape[0] / len(df.index) # row count
df.shape[1] / len(df.index) # column count 


################## Subsetting data - Column & Row Access #########################
iris_df['Sepal.Length'] # series dtype
iris_df[['Sepal.Width', 'Sepal.Length']] # dataframe - double squared brackets if multiple columns
iris_df.iloc[1:10,[1,3]] # using indexes
iris_df.iloc[:, [1:3]]
iris_df.iloc[[1,5,10],[2,4]]
iris_df.iloc[1, np.min(np.where(iris_df.columns == 'Sepal.Width'))] # equivalent to R's which() 
 barb_df.iloc[row,barb_df.columns.get_loc('Channel')] # get_loc more pythonic
iris_df[-10:] # last n rows
iris_df[[col for col in list(iris_df.columns) if 'Sepal' in col]]  # grep approach to column selection 
OG_df[OG_df.columns[~OG_df.columns.isin(OG_df._get_numeric_data().columns)]] # omits unwanted columns  

# Boolean/Logical Subsetting using .loc - stands for LOgical Condition
iris_df.loc[iris_df['Sepal.Length'] > 5.2] 
iris_df.loc[iris_df['Sepal.Length'] < 5.2, ['Sepal.Length', 'Sepal.Width']] 
iris_df.loc[iris_df['Sepal.Length'] < 5.2, iris_df.columns[0:2]] 
iris_df.loc[(iris_df['Species' == 'virginica') | (iris_df['Species'] == 'setosa')]
iris_df[iris_df.Species.isin(['virginica', 'setosa'])] # isin() produces boolean series - can be used to subset rows
iris_df[~ iris_df.Species.isin(['virginica', 'setosa'])] # excludes virgin, setosa i.e. tilda is same as !=

feat_subset.loc[(feat_subset['data_type'] == 'FLOAT64') | (feat_subset['data_type'] == 'INT64'), 'column_name'] ## need parentheses for OR condition 

df_filtered = df.loc[(df>0.8).any(1)]   # any(1) returns any true element over the axis 1 (rows)
df = df['sepal length (cm)'].apply(pd.to_numeric)  # Change the dtype of a column

# To subset using index can simply use df.filter
df_test_unscaled.filter(like='devils', axis=0)

#### To subset data using index value ###
df.loc['623621717403'] # where 623621717403 is the index account number 

# use of query() to filter results - requires boolean param

# subset columns according to dtypes 
df_cat = df.select_dtypes(include = ['O', 'bool', 'category'])
df_num = df.select_dtypes(include = ['float64', 'int64', np.number])
df_numeric = df._get_numeric_data()

# new column #
iris_df['sepalArea'] = iris_df['Sepal.Length'] * iris_df['Sepal.Width']
brics['name_length'] = brics['country'].apply(len) # creates new col which is the length of the char string in another column
iris_df = iris_df.assign(SepalArea = iris_df['Sepal.Length'] * iris_df['Sepal.Width']); iris_df.head()
new_df['no_sports_flag'] = np.where(new_df['base_dt_num'].isin(no_live_sports_base_dt),1,0)    # if want to create new binary column 


df[['Fare']].apply(np.log).plot.box()  # to apply logarithm 
df[['Fare']].apply(lambda x: np.log(x+1)).plot.hist(bins = 20, title = 'Ticket Fare')

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

# Handling Missing Values # fillna(inplace=True) is generally discouraged now - should instead just assign results back to df
iris_df.isna().sum() # missing values per column
iris_df.fillna('missing', inplace = True)    # found that gives SettingWithCopyWarning - ended up using iris_df.loc[:,numeric_cols].fillna(0) / iris_df.loc[:,cat_cols].fillna('Missing)
df[['a', 'b']] = df[['a','b']].fillna(value=0) 
# N.B. Worth noting that using .loc with fillna() and inplace will not work -  won't support inplace operations e.g. train_df.loc[:,['SGE_Active', 'Sports_Active']].fillna(0, inplace=True). Instead..:
# Believe this is because it's dealing with a view rather than a copy. Best bet is to not use inplace and simply assign the result back to the frame, but workaround invovles dict:
df.fillna({'x':0, 'y':0}, inplace=True) # use dictionaries instead - here, x and y are column names
# RECOMMENDATON # 
train_df.loc[:,na_columns] = train_df.loc[:,na_columns].fillna(0)

# Assigning a list to an existing row of a dataframe #
## DO NOT USE THIS APPROACH #
meta_df_all.loc[meta_df_all['sky_abbrev_title'] == k,] = show_results_list # Will not just return row, but will return dataframe too with column names
## USE THIS INSTEAD ##
meta_df_all.loc[meta_df_all['sky_abbrev_title'] == k,meta_df_all.columns.tolist()] = show_results_list # Will work well 



iris_df.dropna() # drop all NaN rows
df.dropna(axis=1) # Drop all columns that contain null values
df.dropna(axis = 1,thresh = n) # Drop all rows have have less than n non null values
iris_df.loc[iris_df['Species'].isna(), 'Region'] = 'Not Applicable'

nullRows = iris_df[iris_df.isnull().any(axis=1)]    # identifies the rows where any of the variables contain a null

iris_df[iris_df.Species.notnull()]
iris_df.iloc[1,2] = np.nan # sets NA 
iris_df.iloc[1:10,2].isnull()
iris_df.iloc[1:10,2].notnull()

# To drop rows from dataframe based on a list of values #
input_dataset[~input_dataset.index.isin(anomalous_accounts)]

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
df.columns = list_of_new_column_names # or just assign list of names
df.rename(columns = {'old_name': 'new_ name'}) # Selective renaming

# Changing dtypes # Can be important for memory efficiency
df.memory_usage(deep=True) #memory usage by column in bytes - will notice that string columns take up way more - so set to categorical if low cardinality 
# With a Categorical, we store each unique name once and use space-efficient integers to know which specific name is used in each row.
# downcast the numeric columns to their smallest types using pandas.to_numeric().
df["id"] = pd.to_numeric(df["id"], downcast="unsigned") # will go from intt64 to int32
df[["x", "y"]] = df[["x", "y"]].apply(pd.to_numeric, downcast="float") # will go from float64 to float32

# One approach to change enmasse
reference_columns = ['show_month','Programme Title','period']
feature_columns = [i for i in final_df.columns.tolist() if i not in reference_columns]
cat_columns = final_df[feature_columns].select_dtypes('O').columns.tolist()
values = ['float64']*len(cat_columns)
col_dtype_dict = dict(zip(cat_columns,values))
final_df = final_df.astype(col_dtype_dict)

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
df.replace(r'^\s*$', np.nan, regex=True) # this replaces whitespace/empty values contained within a dataframe or pandas series

# use of any()
terrestrial_list = ['BBC', 'ITV', 'E4', 'DRAMA', 'CH4', 'PICK', '4MUSIC', 'S4C', '5USA', '5STAR', 'CHANNEL 5', 'FILM4', 'MORE4','4SEVEN']
[i for i in barb_df.Channel.str.upper().unique() if ~any(channel in i for channel in terrestrial_list)]

element_list = [', The', ', A']
[i for i in unique_shows if any(element in i for element in element_list) ]

### BOOLEAN MASK - Mask object is a list of booleans, to be used to subset an array !! ####

# Checking for special chars in string : can help identify which contain these so can remove so can convert from string to numeric 
barb_df['audience'].apply(lambda x: x.isalnum())]


# Working with dates & timestamps # Need to be careful about what you are actually importing here

## String to date #
from datetime import datetime as dt   # differentiates from just import datetime  
barb_df['Date'].apply(lambda x: dt.date(dt.strptime(x,'%d/%m/%Y')))

# date like int 202002182305 - want to create a datetime
df_nonconv['dateHourMinute'].apply(lambda x: datetime.strptime(str(x)+'00', '%Y%m%d%H%M%S'))

# To check timestamp equals some date / tiemstamp
== pd.Timestamp(1900, 1, 1, 8, 30,0)
== datetime.date(1900, 1, 1)

# To change string to date for pandas series
pd.to_datetime(df.Date_feature, dayfirst=True)

dates = pd.date_range("20130101", periods=6)
'YYYY-MM-DD HH:MM:SS' # datetime where supported range is: '1000-01-01 00:00:00' to '9999-12-31 23:59:59'
'1970-01-01 00:00:01' UTC to '2038-01-09 03:14:07' UTC  # supported range for timestamp - Universal Time coordinated - gets converted to current time zone (for that server)
timestamp_col.dt.year # Convert to datetime, before can extract .year/.month/.dayofweek
datetime.today().strftime('%Y-%m-%d') # from datetime import datetime - string format time

datetime.datetime.today().strftime('%Y%m%d_%H%M') # Though might not be aligned to UK time - depends on if BST or GMT 

datetime.strptime('apr' + ' 2021', '%b %Y')  # here %b refers to abbreviated month format, whereas %B refers to the long month format

yesterday = datetime.datetime.combine(
    datetime.datetime.today() - datetime.timedelta(1),
    datetime.datetime.min.time())

# Convert string to date #
date_str = '30-06-2018'
date_object = datetime.strptime(date_str, '%d-%m-%Y').date()

# calculate yyyy-mm-dd format to understand interims 
minBase = new_df['base_dt_num'].min()
maxBase = new_df['base_dt_num'].max()
date_original = datetime.date(1900, 1, 1) # '1900-01-01'   # needs 'import datetime' to work
days_to_add_min = minBase
days_to_add_max = maxBase
date_new_min = date_original + datetime.timedelta(days_to_add_min)
date_new_max = date_original + datetime.timedelta(days_to_add_max)
print(date_new_min)
print(date_new_max)

### TO CHANGE STRING TO TIME FORMAT ####

datetime.datetime.strptime('17:21:60', '%H:%M:%S').time()  ### or barb_df['Start time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').time())
barb_df['start_ts'] = pd.to_datetime(barb_df['Start time'], format='%H:%M:%S')
# Round up to appropriate start & end times
barb_df['start_ts'] = barb_df['start_ts'].dt.round('30min')

#################### JOINS / APPENDING ETC ############
pd.concat([df1, df2], axis = 1) # or axis = 'columns' # R's cbind - order preserved 
iris_df.append(iris_df) # R's rbind()
df1.join(df2, on = col1, how = 'inner')
pd.merge(adf,bdf, how='left' , on='x1')  # how = ['right', 'inner', 'outer']

## ISSUE WITH pd.merge / python in general : trying to join between dates e.g. df['date] must be between df2['date1'] and df2['date2'] - VERY EASY in sql but hard in python!!

# !pip install pandasql
import pandasql as ps

sqlcode = '''
select a.final_offer_proba,
       b.prob_multiplier
from foop AS a
inner join foo AS b ON a.final_offer_proba >= decile_min and a.final_offer_proba < decile_max
'''
newdf = ps.sqldf(sqlcode,locals())



# merge_asof #
import numpy as np
import pandas as pd
foo = pd.DataFrame({'a':[1,5,10], 'b':['oo', 'ah', 'ee']})
foo2 = pd.DataFrame({'a2':[0,8,10], 'b':['oop', 'ahp', 'eep']})
pd.merge_asof(foo, foo2, left_on='a', right_on='a2', direction='nearest')

pd.merge_asof(df1, df2, on='time', direction='nearest') # direction can be forward/backward/nearest -  “backward” selects the last row in the right df whose ‘on’ key is less than or equal to the left’s key   
A “forward” search selects the first row in the right DataFrame whose ‘on’ key is greater than or equal to the left’s key.
A “nearest” search selects the row in the right DataFrame whose ‘on’ key is closest in absolute distance to the left’s key.
pd.merge_asof(df1, df2, on='col', direction='nearest') # similar to a left-join except that we match on nearest key rather than equal keys. Both DataFrames must be sorted by the key
tolerance=pd.Timedelta("5 minutes")) # Can set time tolerance

df3 = pd.merge_asof(left=df1,right=df2,left_index=True,right_index=True,direction='backward',by='account_number',allow_exact_matches=True)

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

probas_dataset.pivot_table(index=probas_dataset.index, columns='discount', values = 'expected_payment') # given that one row represented one account and its expected payment at a single discount level, wanted this rolled up so one row is one account for all discount levels
df.pivot_table(index=col1,values=[col2,col3],aggfunc=mean) | Create a pivot table that groups by col1 and calculates the mean of col2 and col3
df.pivot_table(index = 'Pclass', columns = 'Survived', values = 'PassengerId', aggfunc = 'count') # count of people in each class according to if survived 
df.pivot_table(index = 'Pclass', columns = 'Embarked', values = 'Age', aggfunc = np.mean, fill_value = 0)
df.pivot_table(index = 'Pclass', columns = 'Embarked', values = 'Age', aggfunc = [np.mean, np.median, np.std], fill_value = 0) # MULTIPLE STATISTICS

# Group by to concagtenate multiple rows contraining strings
df.groupby(['name', 'month'], as_index = False).agg({'text_col': ' '.join})
df.groupby(['name', 'month'], as_index = False).agg({'text_col': list}) # concatenates them into a list

df.groupby('SERIES_NAME').agg({'AGREEMENT_TYPE_DESCRIPTION':lambda x: x.nunique()}).reset_index() # count distinct approach for cat columns

## may sometimes want this data in long format so reshape like so :
df2 = df.pivot_table(index = 'Pclass', columns = 'Embarked', values = 'Age', aggfunc = np.median)
stacked = df2.stack() # inverse operation is unstack() 
stacked

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
# A function that behaves like an iterator - it will return the next item
# Returns a generator object (iterable) - to process big data w/o allocating excessive mem simultaneously

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

# n.b. use += in for loop to add incremements - E.G. lottery ticket example : 

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

#### SEQUENCES - All sequences are iterables. Some iterables (things you can `for` loop over) are not sequences (things which you can do x[5] to) e.g. sets/dictionaries.

############### The Asterisk * ###########################
# The asterisk unpacks the iterable object to give the individual elements- just place * left of the object 
foo = iter('Dad'); print(*foo)

# inserts multiple arguments into a list inside the function
def doubler(*sequence):
    return [x*2 for x in sequence]

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

####### ARGS #####
#if you suspect that you may need to use more arguments later on, you can make use of *args
#  accepts a varied amount of non-keyworded arguments within your function
def multiply(*args):
    z = 1
    for num in args:
        z *= num

####### KWARGS #######
# used to pass a keyworded, variable-length argument dictionary to a function
def print_values(**kwargs):
    for key, value in kwargs.items():
        print("The value of {} is {}".format(key, value))

# ** will insert named arguments inside the function as a dictionary:
def arrowify(**args):
    for key, value in args.items():
        print(key+" -> "+value)
arrowify(neutron="n",proton="p",electron="e")

# Can mix the two different approaches: 
def somefunc(a, b, *args, **kwargs):
    print("A:", a)
    print("B:", b)
    print("args:", args)
    print("keyword args", kwargs)
somefunc(1,2,3,4,5, fish = "Haddock")
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

# REVERSE A LIST 
foo = [6,8,9]
foo[::-1]

##################### lambda anonymous functions - functions on the fly #######################

grades = [{'Name':'Jane', 'Score':96}, {'Name':'Mark', 'Score': 102}, {'Name':'Sam', 'Score':98}]
max(grades, key = lambda x: x['Score'])

# Lambda Use Cases # 
# 1. Map Function -  applies the function to all elements in the sequence (iterables) - syntactic sugar for a simple list comprehension that applies one function to every member of a list:
nums = [2,3,4,5]
map (lambda num: num**2, nums) # square all elements - produces map object
lambda nums: map(lambda i: i/2, filter(lambda i: not i%2, nums)) # to get half of all even numbers in a list called nums:

[str(x) for x in range(10)] # or 
list(map(str, range(10)))

df['Curr_Contract_Offer_Length_BB_binary'] = df.Curr_Contract_Offer_Length_BB.map(lambda x :1 if x=='18M' else 0)


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

# To get matrix elementwise multiplicatin:
np.dot(X,Y)


data['Same_Physician'] = data.apply(lambda x: 1 if (x['N_unique_Physicians'] == 1 and x['N_Types_Physicians'] > 1) else 0, axis=1)

names = ['simon', 'mike', 'dorothy']
upper_names = map(str.capitalize, names)
[i for i in upper_names]; # or [*upper_names] to unpack 

nums = [1.4, 8.4, 9.8]
rounded_nums = map(round, nums); print(list(rounded_nums))

### diffeence between apply() and map() and applymap()
# apply() works on a row / column basis of a DataFrame
# applymap works element-wise on a DataFrame (not series)
# map() works element-wise on a Series


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
feat_importance_df[['importances_RF_rank','importances_MI_rank']].apply(np.mean, axis=1) # for rowwise ops - or .apply(lambda x: x.mean(), axis = 1)
iris_df.apply(np.max,axis=0) # axis = 1 for column-wise surely 

########################## List Comprehensions - lapply() equivalent - enables vectorisation ################################

# More efficient than 'for loop's  - Enables vectorisation + parallel processing 
# Can build list comps over all objects except integer objects - in this case, need to create a zip object which is iterable - see below 

[i**2 for i in range(1,10)]
[x/100 for x in range(0,125,25)] # to get a range of floats/non-integers
 
# Conditional list comprehension # 
[ num ** 2 for num in range(10) if num % 2 == 0]
[ num ** 2 if num %2 == 0 else 0 for num in range(10) ]
[f if f in og_cols_list else 'null' for f in feature_selection.final_features.Feature] 
channel_df = df.loc[:, [cc for cc in df.columns if cc.startswith('channel_') and cc.endswith('_mins')]]

# True else False list comp #
boolean_mask = [True if 'only fools and horses' in i else False for i in finished_df.index.tolist()]

# double for 
[x - y for x in range(4) for y in range(4)] # single array generated over all pairs
[[x - y for x in range(4)] for y in range(4)] # nested list comp gives matrix

# Aim is to write code like so in your programs:
analysed_data = [analyze(datum) for datum in data]

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

# Functions can do things to change their mutable arguments, so `return` is optional. #
# However, this is pretty bad practice - functions should normally be side-effect free
def power (number, pow = 2):
    ''' Raise argument to the power of 2, or whatever power is stated '''
    result = number ** pow
    return result 
power(8)

# With side effect, rather than return
def double_inplace(vec):
    vec[:]=[element*2 for element in vec]
z=list(range(4)); double_inplace(z); print(z)

# Return without arguments can be used to exit the function early 
def extend(to, vec, pad):
    if len(vec)>=to:
        return # Exit early, list is already long enough.
    vec[:]=vec+[pad]*(to-len(vec))

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

# To provide type hints: What follows after the arrow indicates the type of the object returned
def splitComma(line: str) -> str:
    ...

######### Sampling ##########
iris_df.sample(frac=0.5)      # randomly selects a fraction of rows 
iris_df.sample(n=10)          # randomly select n rows 

####################### Numpy arrays - uses optimised, pre-compiled C code ###############

# Numpy is more about efficiency, rather than presenting values in a pretty format.
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
matrix2 = np.array([[1,2,4], [5,6,9]]) # list within lists to create a matrix
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

# Numpy shorthand concatentation #
np.c_[np.array(1,2,3),np.array(4,5,6)] # concatentates along the second axis i.e. columnwise join
np.r_[np.array(1,2,3),np.array(4,5,6)] # appends along first axis i.e. appends rowwise


# if an array is empty, then len(array) = 0

cars_array = np.array(['Honda', 'Volvo', 'BMW'])
cars_array = np.append(cars_array, "Renault")
np.where(cars_array == 'Renault')
df[col] = np.where(df[col] == np.inf, 10000, df[col]) # if does not meet condition, will keep value the same as before
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

############################## STRINGS ###########################
%s  # placeholder for a string 
%d  # placeholder for a number (integer)
%f  # Floating point numbers
%   # percentage 
'Hi %s I have %d donuts' %('Alice', 42) # using placeholders

# Using regex .sub for a string
re.sub(r"(:(.*))", "", re.sub(r"(new: )|( season [0-9]+)|( s[0-9]+)|( ep[0-9]+)|( omnibus)|(live: )|(: live)", "", 'Brit Cops: War on Crime'.lower()) ).strip()
# For a pandas series 
barb['Programme Title'] = barb['Programme Title'].apply(lambda x: re.sub(r"(:(.*))", "", re.sub(r"(new: )|( season [0-9]+)|( s[0-9]+)|( ep[0-9]+)|( omnibus)|(live: )|(: live)", "", x.lower() ) ).strip() )

# More regex 
re.match # beginngin gof string must match regex pattern- returns None type if no match
re.search # can exist anywhere in string 
if re.search(r'(\+1)', 'channel4 +1') :  # need escape character for '+' here 
    print('match')

re.sub(r'((^.*?):)','','aws: portrait aw: artist of the year 2021') # removes all characters up to the first colon - Use the question mark to do lazy eval i.e. just stop after finding the first one. ^ says must be at start of string

# Remove punctuation / special characters
re.sub('[^A-Za-z0-9\s]+', '', mystring) # had added \s in too so spaces remain

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

x = 3
if x>0:
    pass  # tells it to do nothing

for n in range(50):
    if n==20: 
        break
    if n % 2 == 0:
        continue
    print(n)

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

# Shuffle examples - Key in ML 
train_df = train_df.reindex(np.random.permutation(train_df.index))

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
a = f.read() # get whole output as a string - vs .readlines() which splits each line using trailing \n
print a
f.close()

# Instead use with() ................This will automatically close files for you even if an exception is raised inside the with block.
#### Context Manager - opens a connection to a file - ensures resources are efficiently allocated when opening a connection to a file. Great as do not need to explicitly close the connection to the file
# creates generator object. Simplifies common resources like file streams - ensures proper acqusiition and release of resources #########

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

with open("file.txt") as file:   
    data = file.read()  

with open("./data/WSJ_02-21.pos", 'r') as f:
    lines = f.readlines()

with open("file.txt", "w") as f:  
    f.write("Hello World!!!") 

# Process very large datasets/files/APIs in chunks, else might put a lof of strain on the RAM 

for chunk in pd.read_csv('rms_data.csv', chunksize=10):
    ....................

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

# Suppress scientific notation # 
final_df['avg_broadcast_reach_3m'] = final_df['avg_broadcast_reach_3m'].apply(lambda x: "{:.0f}".format(x))

# .format()   - uses {} as placeholders 
pubs = ['spoons', 'swan', 'jimmys']  
drinks = ['beer', 'shandy', 'wine']
for p in pubs: 
    for d in drinks:
        print('I love drinking {} at {}'.format(d,p))

# Or can use
print(f'Initial string:  {data}')

print("Model Accuray: {:.2f}%".format(100*classifier.score(X_test, y_test))) # Model Accuray: 82.58%

# can even use keyword placeholders #
print('Hello {name}, {greeting}'.format(greeting = 'Goodmorning', name = 'John'))

# Alternative way to .format() is using % as per the c programming style
x = 12.3456789
print('The value of x is %3.2f' %x)

print('Proportion of reelgood scores missing is {:.2f}%'.format(gt_df.reelgood_score.isna().sum() / gt_df.shape[0]) ) #2 decimal places

# ord() gets ASCII code for character e.g. 'a' is 97 #
ord('$') # returns integer that represents the Unicode code point for the given Unicode character

# Progress bar # 
from tqdm import tqdm

from tqdm import notebook
notebook.tqdm().pandas() # means all pandas DataFrams now have new methods - prefix 'progress_' which allows you to see how long apply and applymap will take 
df.progress_apply() ......__annotations__

#### EXCEL EXPORTING #####

.to_clipboard(index=False) # then use Ctrl+V once in excel 

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


####### STORING MODELS - involves seralising the model to save it, before deserialising it to load it in for use
# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
# N.B. When deserialising a model, will need same python version/library versions - not limited to just sklearn/numpy

## Pickling - import pickle #  The pickle API for serialising standard Python objects : e.g. trained models ## 
# For serializing and de-serializing a Python object structure
# “Pickling” (i.e. serialisation/flattening) is the process whereby a Python object hierarchy is converted into a byte stream
# “unpickling” is the inverse operation, whereby a byte stream is converted back into an object hierarchy - ensure legitimate source before unpickling file
# cPickle module - optimised as written in c but has other issues .....
import pickle 
# store it to file
clf = RandomForestClassifier().fit(X_train, y_train)
with open('model.pkl', 'wb') as file:
    pickle.dump(clf, file = file)
# to load it from file
with open('model.pkl', 'rb') as file:
    clf2 = pickle.load(file)

foo = pickle.dump(clf_model)
clf_model2 = pickle.load(foo)
clf_model2.predict(X_train)

#### E.G. 
# Load the current model from disk
champion = pickle.load(open('model.pkl', 'rb'))
# Fit a Gaussian Naive Bayes to the training data
challenger = GaussianNB().fit(X_train, y_train)
# Print the F1 test scores of both champion and challenger
print(f1_score(y_test, champion.predict(X_test)))
print(f1_score(y_test, challenger.predict(X_test)))
# Write back to disk the best-performing model
with open('model.pkl', 'wb') as file:
    pickle.dump(champion, file=file)  # as champion got higher F1 score 


## import joblib - similar to pickle; joblib API efficiently serialises objects with numpy arrays ##
# provides utilities for saving & loading python objects that make use of Numpy data structures, efficiently. 
# useful for ML algos that require a lot of params or store the entire dataset (think KNN)

from sklearn.externals import joblib
joblib.dump(my_model, "my_model.pkl")
# and later...
my_model_loaded = joblib.load("my_model.pkl")

###### What about dill ? ### Idea behind it is hetereogenous computing ### 
# Extends pickle module for serializing and de-serializing python objects - provides same interface as pickle 
# But also includes some additional features to pickle - can save the state of an interpreter session in a single command
# Could save interpreter session, close the interpreter, ship the pickled file to another computer, open a new interpreter, unpickle the session 
# and thus continue from the ‘saved’ state of the original interpreter session
# Can be used to store python objects to a file, but the primary usage is to send python objects across the network as a byte stream
# Allows arbitrary user defined classes and functions to be serialized. Not intended to be secure against erroneously or maliciously constructed data

###### LOGGING - never log actual data - just messages ######
# Often will just print logging to console, but can create a logging file and read if using fileConfig()
import logging
# getLogger() provides reference to a logger instance with the specified name if provided (or root if not)
# Loggers that are further down in the hierarchical list are children of loggers higher up in the list e.g. if logger with name foo (getLogger('foo'))
# then  loggers with names of foo.bar, foo.bar.baz, and foo.bam are all descendants of foo

logger = logging.getLogger('simple_example') # create logger
logger.setLevel(logging.DEBUG) # logging.CRITICAL / ERROR / INFO / NOTSET also exist - determines level of verbosity 

logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')


## How to create a logging file - see https://docs.python.org/dev/library/logging.html#logging.basicConfig

# N.B. Can set different levels of printing for each handler type - so might be DEBUG for file handler while WARNING for StreamHandler
import logging
def create_logger(log_file_name=None):
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    if log_file_name is not None:
        # Sets up logging to print to a file
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S',
                            handlers=[
                                logging.FileHandler(log_file_name, mode='w'), # sets up logging to a file
                                logging.StreamHandler() # sets up logging to console i.e. printed to console directly
                                ]
                                )
    else:
        logging.basicConfig(format='%(asctime)s - %(message)s',datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger('TreeRulesExtractor')
    logger.setLevel(logging.INFO)
    print = logger.info
    return logger


###### YAML ######
with open('test.yaml') as file:
    yaml_contents = yaml.load(file, Loader=yaml.FullLoader)
    print(yaml_contents)
# yaml.load() vs yaml.safe_load() 
# believe yaml.safe_load() == yaml.load(Loader=yaml.SafeLoader) # If your yaml data is coming from trusted sources, you can specify the Loader=FullLoader argument which loads the full YAML language and avoid the arbitrary code execution
# safe_load() - to load data by untrusted source

####### WRITING TESTS FOR YOUR CODE ################

# Separate python scripts for testing : e.g. tests.py / test_sample.py

# Option 1 : pytest #

# Option 2 : unittest #

# Option 3 : tox # 
pip install tox

# ASSERT - used when debugging code # 
x = 'hello'; assert x == 'hello' # if condition holds then nothing happens
assert x == 'goodbye', 'x should be hello'
assert x == 'goodbye' # if condition does not hold AssertionError is raised


##### Ipython Notebooks ####

# using cell magics 
%%bash  # executes this cell as *shell code*
python -c "print(2*4)"

%matplotlib inline # renders graphic in NB instead of just dumping the data into a variable. Also, suppresses output of the function on final line.  Don't forget to add a semicolon to suppress the output and to just give back the plot itself. 
############

%%bash
echo '#!/usr/bin/env python' > eight
echo "print(2*4)" >> eight
chmod u+x eight
./eight


#### yield vs return in your function ####
# while return would immediately enter the function after returning the values, yield returns the values and simply pauses the function. E.G.
# yield is a generator function - will continue running if more values are needed 
def get_windows():
    i = C
    while i < len(words) - C:
        center_word = words[i]
        context_words = words[(i - C):i] + words[(i+1):(i+c+1)]
        yield context_words, center_word
        i += 1


###### Write library/package to disc - which can then be imported for use in a program 
%%writefile draw_eight.py 
# Above line tells the notebook to treat the rest of this
# cell as content for a file on disk.

import numpy as np; import math; import matplotlib.pyplot as plt
def make_figure():
  theta=np.arange(0, 4 * math.pi, 0.1)
  eight=plt.figure()
  axes=eight.add_axes([0, 0, 1, 1])
  axes.plot(0.5 * np.sin(theta), np.cos(theta / 2))
  return eight
################# Could then import : import draw_eight.py


##### Memory & References ###### http://www.pythontutor.com/visualize.html#mode=display 
Each object requires some computer memory (an address in your comp)
Each label (variable) is a reference to such a place - can be called the (Global) Frame
If an object no longer has any labels/variables, then this object can no longer be found - Pythons Garbage collector will remove this data
This Avoids memory addresses without references taking up more memory

# E.G. 
name = 'James' # here, James is the string object
name = 'Jim' # now name is the label that refers to the string object 'Jim' - this means that 'James' no longer has a reference/label so the garbage collector should remove it 

# E.G. 2 - TRY THIS IN URL ABOVE FOR VIZ PURPOSES
list1 = [1,2,4,5]
list1.append(1)

# E.G. 3 See how nested containers look
x = [ ['a', 'b'] , 'c']

# Identity vs equality - Having the same data is different from being the same actual object in memory
[1, 2] == [1, 2]
[1, 2] is [1, 2]
# == checks, element by element, that the two containers have the same data VS `is` operator checks that they are actually the same object#
# BUT BEWARE OF THE SUBTLETY ... for immutable the python language might save memory by reusing a single instantiated copy. This will always be safe.
"Hello" is "Hello"
# e.g. 2
x = range(3); y=x; z=x[:]
x == y; x is y

#### create new feature - defining the number of unique protocols per source computer - then join back to original df
protocols = flows.groupby('source_computer').apply(
  lambda df: len(set(df['protocol'])))
protocols_DF = pd.DataFrame(
  protocols, index=protocols.index, columns=['protocol'])
X_more = pd.concat([X, protocols_DF], axis=1)


## Module (file or collection of files) vs package (directory of modules) ##
## Decorators ## 


##### subprocess module ########### intended to replace functions such as os.system(), os.spawn*(), os.popen*(), popen2.*() and commands.*()
# To execute programs written in different languages (e.g. C, Java, Shell etc), through python code by creating new processes (i.e. to spawn new processes)
# i.e. to execute external commands
# Plus, helps obtain the input/output/error pipes/streams as well as the exit codes of various command
# N.B. Avoid running shell=True - security hazard if dealing with untrusted inputs
# Helps easily integrate shell command into scripts w/o need to pop in and out of script
import subprocess
# N.B. Believe should use subprocess.run for python version 3.5+ rather than subprocess.call() as it's safer 
subprocess.call(['ls', '-1'], shell=True) # command line arguments are passed as a list of strings, which avoids the need for escaping quotes or other special characters that might be interpreted by the shell
subprocess.call('echo $HOME', shell=True)
subprocess.run(["cat", "data.txt"])
subprocess.run(["cat", "data.txt"], stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) # to suppress any form of output
process = subprocess.run(['ls','-lha'], check=True, stdout=subprocess.PIPE, universal_newlines=True); process.stdout; process.stderr; process.returncode # universal_newlines - ensures string not bytes

output = subprocess.run(["cat", "data.txt"], capture_output=True); print(output); print(output.stdout); print(output.stderr) # output produced as byte sequence - so alternatively use  print(output.stdout.decode(“utf-8”)) OR subprocess.run(['cat', 'data.txt'], text=True)
# To run a command directly into a shell as is, must pass “shell=True”  - discouraged by python developers as using “shell=True” can lead to security issues e.g. code injection if you use user input to build the command string
subprocess.run("cat 'data.txt’", shell=True)
# To tokenise simple shell commands:
import shlex; shlex.split('ls -l') 
# popen - used for more complex examples than sp.run()
from subprocess import Popen
p = Popen(["ls","-lha"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True); output, errors = p.communicate()
# or
p = subprocess.Popen(["echo", "hello world"], stdout=subprocess.PIPE); print(p.communicate()) # The communicate() method returns a tuple (stdoutdata, stderrdata).




### defaultdict

# special kind of dictionary that returns the "zero" value of a type if you try to access a key that does not exist e.g. if defaultdict(int) as want a dictionary containing int values, then would return 0 if no key found
word_count_dict = defaultdict(int)
for word in corpus:
    word_count_dict[word] += 1
word_count_dict['ginormous'] # will return 0 even though not a word in the corpus - as opposed to an error 



########## CURVE FITTING ##########

# Use logit as bounded between 0-1 # 
from scipy.optimize import curve_fit
def logit(x, b0, b1):
    exp_comp = np.exp(-(b0+b1*x))
    return 1/(1+exp_comp)
pars, _ = curve_fit(f=logit, xdata=x_dummy, ydata=y_dummy)
