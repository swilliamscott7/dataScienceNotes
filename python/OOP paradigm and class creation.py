##### OOP ###########

### Fundamentals of OOP include : composition / inheritance / polymorphism ###

### Class = reusable chunk of code that has associated methods and attributes
# The class is a cookie cutter, while the object is an instance of the class i.e. the cookie created from it 
# The class performs a logical group of tasks
# Class is a template/blueprint to be reused to make many objects with properties based off the parent class 
# Constructors - used for instantiating an object; think of it as __init__ialising a class 
# each method takes self as a parameter
# each attribute is self. something e.g. self.dim, self.shape, self.summary
# each object has attributes and methods 

# What is self ? self represents the instance of the class. Think of self as the 


class Trex():  # declares a class - don't need to fill this with anything 
    pass       # means we're not putting any context/values into the class yet
Trex_1 = Trex() # Trex_1 is an instance of the class - has variables & methods

# Empty constructor 
class Dinosaur: 
    def __init__(self):
        pass
    
# Constructor with attributes (i.e. variables)
class Dinosaur: 
    def __init__(self): 
        self.tail = 'Yes'    # means every object of the dinosaur class, will have a tail 

class Math:
    def __init__(self, x, y):   # the __init__ function is the constructor - is a special method
        self.x = x
        self.y = y
    def add(self):
        return self.x + self.y
    def subtract(self):
        return self.x - self.y
    
class AdvancedMath:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def multiply(self):
        return self.x * self.y
    def divide(self):
        return self.x / self.y
    
class GeekforGeeks: 
  def __init__(self): 
      self.geek = "GeekforGeeks" # A default constructor 
        
class DataShell: 
    def __init__(self, filename): # constructor 
        self.filename = filename  # the class variable/attribute
    def create_datashell(self) # a method
        self.array = np.genfromtxt(self.filename, delimite=',', dtype=None) 
        return self.array
    def rename_column(self, old_colname, new_colname): # a second method
        for index, value in enumerate(self.array[0]):
            if value == old_colname.encode('UTF-8'):
                self.array[0][index] = new_colname
        
# identify if object is an instance of a class using isinstance() # 
class myObj:
  name = "John"
y = myObj()
x = isinstance(y, myObj) # boolean output
x
y.name


class DataShell:
    def __init__(self, identifier, data):
        self.identifier = identifier # instance attribute 1
        self.data = data # instance attribute 2
x = 100; y = [1, 2, 3, 4, 5]
my_data_shell = DataShell(x, y); print(my_data_shell.identifier); print(my_data_shell.data)

# Static Variables - are consistent across instances #
class Dinosaur():
    eyes = 2 # static variable - means all instances will have 2 eyes, no matter the arguments passed through the instance
    def __init__(self, teeth):
        self.teeth = teeth     # example of an instance variable - changes according to argument passed through
stegosaurus = Dinosaur(40)
stegosaurus.teeth = 40; stegosaurus.teeth; stegosaurus.eyes
# can override static variables like so :
stegosaurus.eyes = 1



class DataShell:

    def __init__(self, dataList):
        self.data = dataList # Declare data as an instance variable and assign it the value of dataList.
        
    def show(self):
        print(self.data)
        
    def avg(self):
        avg = sum(self.data)/float(len(self.data))
        print(avg)

my_data_shell = DataShell([x for x in range(1,10)])
my_data_shell.show()
my_data_shell.avg()

class DataShell:

    def __init__(self, filepath):
        self.filepath = filepath
        self.data_as_csv = pd.read_csv(filepath)

    def rename_column(self, column_name, new_column_name):
        self.data_as_csv.columns = self.data_as_csv.columns.str.replace(column_name, new_column_name)

    def get_stats(self):
        return self.data_as_csv.describe()
    
us_data_shell = DataShell(us_life_expectancy)
us_data_shell.rename_column('code', 'country_code')
print(us_data_shell.get_stats())


###############  Inheritance ###############

# A class that inherits attributes from another 'parent class' and extends on it to add more unique functionality 
# Analogy : Use the dinosaur cookie cutter attributes as a base for a new pterodactyl class - Class pterodactyl(dinosaur)
# Can identify if class inherited using e.g. is a ptero a dinosaur  

#### Inheritance ####
class StDevDataShell(DataShell): # this class inherits all attributes and methods from DataShell
    pass 

class Animal:
    def __init__(self, name):
        self.name = name

class Mammal(Animal):
    def __init__(self, name, animal_type):
        self.animal_type = animal_type

class Reptile(Animal):
    def __init__(self, name, animal_type):
        self.animal_type = animal_type

daisy = Mammal('Daisy', 'dog'); print(daisy)
stella = Reptile('Stella', 'alligator'); print(stella)

## Polymorphism - where two classes are inherited from a parent class but with differences
class Vertebrate:
    spinal_chord = True
    def __init__(self, name):
        self.name = name

class Mammal(Vertebrate):
    def __init__(self, name, animal_type):
        self.animal_type = animal_type
        self.temperature_regulation = True

# Create a class Reptile, which also inherits from Vertebrate
class Reptile(Vertebrate):
    def __init__(self, name, animal_type):
        self.animal_type = animal_type
        self.temperature_regulation = False

daisy = Mammal('Daisy', 'dog')
stella = Reptile('Stella', 'alligator')
print("Stella Spinal cord: " + str(stella.spinal_chord))
print("Stella temperature regularization: " + str(stella.temperature_regulation))
print("Daisy Spinal cord: " + str(daisy.spinal_chord))
print("Daisy temperature regularization: " + str(daisy.temperature_regulation))



################## Composition ##########################

# Classes can be inherited from other classes + can also be made up of other classes 
# It involves taking elements of several different classes to create a kind of 'frankenstein' class 
# Analogy - to make a lochness monster, we don't have a cookie cutter template, so take a neck from the giraffe class, the tail from the whale class, and the teeth from the t-rex class

class DataShell:
    family = 'DataShell' # class variable
    def __init__(self, name, filepath): 
        self.name = name
        self.filepath = filepath

class CsvDataShell(DataShell):
    def __init__(self, name, filepath):
        self.data = pd.read_csv(filepath)
        self.stats = self.data.describe()

class TsvDataShell(DataShell):
    def __init__(self, name, filepath):
        self.data = pd.read_table(filepath)
        self.stats = self.data.describe()

us_data_shell = CsvDataShell("US", us_life_expectancy); print(us_data_shell.stats)
france_data_shell = TsvDataShell('France', france_life_expectancy); print(france_data_shell.stats)