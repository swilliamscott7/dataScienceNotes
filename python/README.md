## INSTALLATION ##

- Pip vs Conda
  - pip is default package manager for python
  - conda handles a lot of dependencies for you + easily creation of environments + focussed around DS  
  -
1. Install directly or using a distributor like conda
2. Conda etc. will create the root environment for python with e.g. python 3.8
3. Can then install other python packages within this root environment 
4. Later can add as many python environments with different python version and different packages within each environment  

Virtual Environments?
  - use virtualenv
  - should always use either virtual environments or Docker containers for working with Python
  - Handles dependencies
  - Can ensure your app will work on a client computer 
  - Helps colleagues collaborate 
  
  - Using Conda to create virutal env
    - $ conda create -n  tensorflow-env  python=3.6   # n.b tensorflow does not work with 3.7 atm
    - $ conda create --name myenvname python=3.4  # e.g.3 conda create -y --name tensorflow python=3.6
    - $ conda activate tensorflow-env    # activates the new environment
    - $ conda install tensorflow 
    - $ conda env remove --name tensorflow-env  # removes environment
  -  How to create a new environment, but inherit packages from another environment:
    - $ conda create -n your-env-name --clone base   # where base is the environment
    
    - think to add it as a kernel:
      $ conda install ipykernel
      $ python -m ipykernel install --user --name=env_name
  
# isolating dependency management on a per-project basis will give us more certainty and reproducibility than Python offers out of the box



Useful design patterns in python:
- Factory pattern
- Decorator Pattern (e.g. @property,@classmethod,@staticmethod)
- The Building pattern
