# Cell Cycle
***

This is an implementation of : Non invasive live cell cycle monitoring using a supervised deep neural autoencoder onquantitative phase images
  
When using this code , please cite Barlaud, M., Guyard, F.: Learning sparse deep neural networks 
using efficient structured projections on convex constraints for green ai. ICPR 2020 Milan Italy (2020)

and 

Philippe Pognonec, Axel Gustovic, Zied Djabari, Thierry Pourcher and Michel Barlaud : Non invasive live cell cycle monitoring using a
supervised deep neural autoencoder on quantitative phase images


## Table of Contents
***
1. [Installation](#installation)
2. [How to use](#use)
  
    
## Installation : 
***

First, you will need a python runtime environment. If you don't have one on your computer we recommend you to download anaconda (https://www.anaconda.com/products/individual#Downloads). It is a platform that brings together several IDEs for machine learning. In the rest of this tutorial we will use Spyder. 
You can now download the code in zip format and unzip it on your computer.
Then, to execute our script, we will need several dependencies. To install them you will have to run this command in the spyder console (at the bottom right).
```
$ conda install -c anaconda pip
$ cd path/to/project
$ pip install -r requirements.txt (Warning, before launching this command you must go to the directory where the requirements.txt is located)
```

To install captum make sure you have a c++ compiler

## How to use : 

Everything is ready, now you have to open the code in spyder (top left button). 
Then run it with the Run files button. It is possible to change the parameters and the database studied directly in the code. 

Here is a list of modifiable parameters with our values : 

| Parameters | line in code | recommended value |
|:--------------|:-------------:|--------------:|
| ETA | 119 | 600 |
| Seed | 43 | 5 |
| Database | 70 | - |
| Number of Control cells in training | 74 | 1000 |
| Number of Control cells in the test  | 75 | 5000 |
