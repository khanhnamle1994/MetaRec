Welcome to the research code repository for my Master's Thesis work on Deep Learning Based Recommendation Systems. This work is still in progress.

![header-image](https://miro.medium.com/max/1400/1*lJhKb5Rl47RSrwI8mVB_5g.png)

# Background

Recommendation systems are technologies and techniques that can provide recommendations for items to be of use to a user.
The recommendations provided are aimed at supporting their users in various decision-making processes, such as what products to purchase, what music to listen, or what routes to take.
Correspondingly, various techniques for recommendation generation have been proposed and deployed in commercial environments.
The goal of this research is to impose a degree of order upon this diversity by presenting a coherent and unified repository of the most common recommendation methods to solve the collaborative filtering problem:
from classic matrix factorization to cutting-edge deep neural networks.

# Datasets

For my experiments thus far, I worked with the [MovieLens1M Dataset](https://github.com/khanhnamle1994/transfer-rec/tree/master/ml-1m), a famous dataset within the recommendation systems research community.
The data contains 1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users who joined MovieLens in 2000.

Here are other datasets that I plan to experiment with:
- [ ] [Spotify RecSys 2018 Challenge](http://www.recsyschallenge.com/2018/)
- [ ] [Trivago RecSys 2019 Challenge](http://www.recsyschallenge.com/2019/)
- [ ] [Twitter RecSys 2020 Challenge](http://recsys-twitter.com/)

# Research Models

## [Matrix Factorization](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments)

Here are the 7 different Matrix Factorization models for Collaborative Filtering:

* [Vanilla Matrix Factorization](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/Vanilla-MF)
* [Matrix Factorization with Biases](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/MF-Biases)
* [Matrix Factorization with Side Features](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/MF-Side-Features)
* [Matrix Factorization with Temporal Features](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/MF-Temporal-Features)
* [Factorization Machines](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/Factorization-Machines)
* [Matrix Factorization with Mixture of Tastes](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/MF-Mixture-Tastes)
* [Variational Matrix Factorization](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/Variational-MF)

## [Multi-Layer Perceptron](https://github.com/khanhnamle1994/transfer-rec/tree/master/Multilayer-Perceptron-Experiments)

Here are the 5 different Multilayer Perceptron models for Collaborative Filtering:

* [Wide and Deep Learning](https://github.com/khanhnamle1994/transfer-rec/tree/master/Multilayer-Perceptron-Experiments/Wide-and-Deep-PyTorch) ([paper](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Wide-and-Deep-Learning-for-Recommendation-Systems.pdf))
* [Deep Factorization Machines](https://github.com/khanhnamle1994/transfer-rec/tree/master/Multilayer-Perceptron-Experiments/DeepFM-PyTorch) ([paper](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/DeepFM-A-Factorization-Machine-Based-Neural-Network-For-CTR-Prediction.pdf))
* [Extreme Deep Factorization Machines](https://github.com/khanhnamle1994/transfer-rec/tree/master/Multilayer-Perceptron-Experiments/xDeepFM-PyTorch) ([paper](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/xDeepFM-Combining-Explicit-and-Implicit-Feature-Interactions-For-Recommender-Systems.pdf))
* [Neural Factorization Machines](https://github.com/khanhnamle1994/transfer-rec/tree/master/Multilayer-Perceptron-Experiments/Neural-FM-PyTorch) ([paper](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Neural-Factorization-Machines-For-Sparse-Predictive-Analytics.pdf))
* [Neural Collaborative Filtering](https://github.com/khanhnamle1994/transfer-rec/tree/master/Multilayer-Perceptron-Experiments/Neural-CF-PyTorch-Version2) ([paper](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Neural-Collaborative-Filtering.pdf))

## [Autoencoders](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments)

Here are the 6 different Autoencoders models for Collaborative Filtering:

* [AutoRec](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/AutoRec-TensorFlow) ([paper](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/AutoRec-Autoencoders-Meet-Collaborative-Filtering.pdf))
* [DeepRec](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/NVIDIA-DeepRec-TensorFlow) ([paper](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/Training-Deep-Autoencoders-For-Collaborative-Filtering.pdf))
* [Collaborative Denoising Autoencoders](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/CDAE-PyTorch) ([paper](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/Collaborative-Denoising-Autoencoders-for-TopN-Recommendation-System.pdf))
* [Multinomial Variational Autoencoders](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/VAE-PyTorch) ([paper](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/Variational-Autoencoders-for-Collaborative-Filtering.pdf))
* [Sequential Variational Autoencoders](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/SVAE-PyTorch) ([paper](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/Sequential-Variational-Autoencoders-for-Collaborative-Filtering.pdf))
* [Embarrassingly Shallow Autoencoders](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/ESAE-PyTorch) ([paper](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/Embarrassingly-Shallow-Autoencoders-for-Sparse-Data.pdf))

## [Boltzmann Machines](https://github.com/khanhnamle1994/transfer-rec/tree/master/Boltzmann-Machines-Experiments)

Here are the different Boltzmann Machines models for Collaborative Filtering:

* [Restricted Boltzmann Machines](https://github.com/khanhnamle1994/transfer-rec/tree/master/Boltzmann-Machines-Experiments/RBM-CF-PyTorch)

# Production App

## [Django Web Service](https://github.com/khanhnamle1994/transfer-rec/tree/master/Django-Web-Service)

Here I built a recommendation web service with Python 3.6 and Django 2.2.4. It has these properties:
- Can handle many API endpoints,
- Each API endpoint can have several research algorithms with different versions,
- Research code and artifacts (files with model parameters) are stored in the code repository (git),
- Supports fast deployments and continuous integration (tests for both: server and research code),
- Supports monitoring and algorithm diagnostic (support A/B tests),
- Is scalable (deployed with containers),
- Has a user interface.

# Blog Posts

I have written a series of blog posts documenting my experiments on [my website](https://jameskle.com/writes/category/Recommendation+System):
- [Part 1: An Executive Guide to Building Recommendation System](https://jameskle.com/writes/rec-sys-part-1)
- [Part 2: The 10 Categories of Deep Recommendation Systems That Academic Researchers Should Pay Attention To](https://jameskle.com/writes/rec-sys-part-2)
- [Part 3: The 6 Research Directions of Deep Recommendation Systems That Will Change The Game](https://jameskle.com/writes/rec-sys-part-3)
- [Part 4: The 7 Variants of Matrix Factorization for Collaborative Filtering](https://jameskle.com/writes/rec-sys-part-4)
- [Part 5: The 5 Variants of Multi-Layer Perceptron for Collaborative Filtering](https://jameskle.com/writes/rec-sys-part-5)
- [Part 6: The 6 Variants of Autoencoders for Collaborative Filtering](https://jameskle.com/writes/rec-sys-part-6)