## Overview

For our semester-length project in machine learning, we analyzed and built models for a dataset from Kaggle – the Old Bailey dataset. “The Old Bailey” is the name for a court that mainly operates in London and has been keeping records of its court proceedings and outcomes (guilty/not guilty) since the 16th century. The goal of the project was to create classifiers/models that can correctly determine the outcome of a case based on different datasets created from the court proceedings. The proceedings themselves were preprocessed into 4 datasets: bag-of-words (BOW), “term frequency-inverse document frequency” or tfidf, glove, and miscellaneous. Without going into too much detail, bag of words counts the words, tfidf weights frequent words lower, glove represents the document by the average of its “word embeddings”, and miscellaneous is a set of categorical attributes extracted from the trial data. A main difference between these sets is the density of data- for instance, the glove set is 300 features and is densely populated with data, whereas the BOW and tfidf set are 10,000 features and contain many empty/zero values for many of those features in each example. We were given each of these datasets, separated into training, testing, and evaluation slices, and left largely to our own discretion as to what combination of classifiers, datasets, and even processing/combination of the data we selected to try and get the highest accuracy possible.

## Main Ideas Explored

### ID3 (Trees)

The first algorithm I attempted to use to classify the data was the ID3 algorithm. The ID3 algorithm is a batch learning algorithm in that it uses the whole dataset once to build the tree, rather than learning from / making updates to itself from mistakes. The main idea is to build the tree by putting the attributes that tell us the most about the dataset (information gain) at the top, and the less information-gaining attributes at the bottom. The trees must use categorical attributes and tend to work better with fewer features and more densely populated datasets so I first tried the glove dataset as it only had 300 as opposed to 10,000 features. The glove set had continuous data, so I discretized the values to values between 1 and 10. I also tried it on the miscellaneous set later, which I figured should work pretty well as it only has 6 attributes of all categorical data.

### Perceptron

The next algorithm I used was the perceptron algorithm. The perceptron algorithm aims to create a decision boundary that separates the two types of labels guilty and not guilty- 1 and 0. Unlike ID3, perceptron is a supervised learning algorithm in that it updates a weight vector and a bias term based on when it makes errors. Since the model works pretty easily with numerical data, continuous or discrete, I ran it on all the data sets except the miscellaneous set. Since I ran it on all 3, I was able to explore how the classifier performed on high dimensionality set of varying densities, as well as a super-dense low dimensionality set. I didn’t transform the data at all for these tests.

### SVM / SVM Logistic

The SVM (support vector machine) is another supervised learning algorithm that is supposed to work particularly well in dealing with high dimensionality datasets. It also continuously updates a weight vector and a bias term based on mistakes and tries to maximize the margin between its classifier and the example closest to it, giving better generalization and resistance to noisy data. I also used this model on all datasets except miscellaneous. More specifically, I implemented stochastic sub-gradient descent for SVM which iterates over the entire dataset at each epoch and updates the weights and bias based on error. I also implemented a similar SVM that used logistic regression as its objective function, this time with stochastic gradient descent. This version of gradient descent calculated a gradient using a single example at each epoch and updated the weights based on that gradient and learning rate.

### SVM over Trees

SVM over trees was the last idea I tried for this project, as I never did get the logistic SVM working very well. This implementation built an ensemble of depth-limited decision tree predictions over the data that was then to be learned by the SVM. I constructed 100 trees based on randomly selected 1000 examples for each tree, then used those to transform the test, train, and eval datasets of the miscellaneous dataset by giving each tree's prediction for each example- effectively making the dataset 100 dimensional. I then trained and tested my stochastic sub-gradient descent SVM on these transformed sets. I used the miscellaneous set because it had the best accuracy I had tested thus far for tree classifiers.
