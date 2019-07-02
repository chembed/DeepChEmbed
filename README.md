[![Build Status](https://travis-ci.org/chembed/DeepChEmbed.svg?branch=master)](https://travis-ci.org/chembed/DeepChEmbed)

# DeepChEmbed
Training domain-aware embeddings for clustering purposes, written in Python and Keras  

Authors: **Hang Hu**, **Yang Liu**, **Yueyang Chen**

----

## Overview  

__DeepChEmbed__  is an open-source python package which develops new types of chemical embeddings for the purpose of improving the classification of chemical properties, such as biodegradability, toxicity and _etc_.

----

### Current Features  

* Wrapper model class for Kmeans and Autoencoder
* Combined training of Autoencoder and KMeans Clustering/Classifying 
* Wrapper function for visualization of high dimensional data using t-SNE projection

----

### Incoming Features

* Coupling with advanced autoencoding method, such as convolutional autoencoder 
* Coupling with other classification algorithms, such as support vector machines, etc.
* Developing ?interpretable? embeddings: cooperated with the chemical meanings

----

## Getting Started to train your own model

### Pre-requirements:

* Python, version 3.6.7 or later
* Conda, version 4.6.8 or later
* Numpy, version 1.16.3 or later
* Pandas, version 2.2.4 or later
* Keras, version 1.16.3 or later
* Tensorflow, version 1.13.1 or later
* Scikit-learn, version 0.20.3 or later
* Matplotlib, version 0.9.0 or later
* Seaborn, version 1.16.3 or later
* RDKit, version 2019.03.1 or later
* Mordred, version 1.1.1or late

### Environment Installation

 You can execute the following ``commands`` from your computer's terminal application: 
 
 1. Either clone the _deepchembed_ repository:

    ``git clone https://github.com/chembed/deepchembed.git ``  

    or download the zip file:  

    `` curl -O https://github.com/chembed/deepchembed/archive/master.zip ``
 
2. `` cd deepchembed ``

2. ``conda env create -n environment.yml``

3. ``conda activate deepchembed``

### 	Tutorials

You can find all the tutorial scripts in this directory: [https://github.com/chembed/DeepChEmbed/tree/master/deepchembed/tutorials](https://github.com/chembed/DeepChEmbed/tree/master/deepchembed/tutorials)

## Directory Structure

       deepchembed (master)
    |--data  
       |-- 
    |--doc  
       |-- 
    |--deepchembed
       |--notebook_scripts
         |---
    |--tutorials
       |--
    |--tests
       |--
       |--\_\_init\_\_.py
       |--cluster.py
       |--dce.py
       |--descriptor.py
       |--dimreducer.py
       |--utilities.py
    |--.coverage
    |--.gitignore
    |--.travis.yml
    |--LICENSE 
    |--README.md
    |--environment.yml
    |--requirements.txt


## Contributions

Any contributions to the project are warmly welcomed! If you discover any bugs, please report them in the [issues section](https://github.com/chembed/deepchembed/issues) of this repository and we'll work to sort them out as soon as possible. If you have data that you think will be good to train our model on, please contact one of the authors. 


## License

_deepchembed_ is licensed under the [MIT license](https://github.com/chembed/deepchembed/blob/master/LICENSE).
