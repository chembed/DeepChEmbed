# Use Cases  

## Background  
Clustering a set of molecules into different groups based on their molecular properties is very useful to establish   
understanding of how features are related to the separation. Good separation of chemical property is fundamental   
to data reprentation and visualization.   
Performances of common clustering methods is highly dependent on the input data. Dimensionality reduction (DR) is  
often applied to map the high-dimensional data into clustering-friendly space to allow for easier separation. Compared  
to linear DR methods (e.g. PCA, CCA), deep neural networks has demonstrated its power in nonlinear mapping of high-  
dimensional domain into reduced-dimensional latent spaces. The latent representation of data can then be used as  
input for clustering tasks.  
Combining deep neural network (DNN) based DR and K-means clustering would require a two-phase training procedure  
which optimize on both the representation learning and the clustering. 


## Objectives  

Achieve robust and efficient clustering of a large unlabled datasets (e.g, ChEMBL with 1.8M chemicals) by:
* 1. Explore autoencoder neural network to reduce dimension of the given dataset
* 2. Adapt K-mean clustering algorithm to the latent space
* 3. Evaluate the clustering result by the loss function of the autoencoder, the convergence of K-mean method  
     and the compatibility between the autoencoder and the k-mean method.

## Components  

### Data Cleaning  

### Embedding Layers

### Clustering  

### Validation  

1. Smaller synthetic dataset   
A small synthetic dataset that contains 100+ of feature and 1000+ of data points is used to evaluate

2. Larger unlabeled dataset


