import numpy as np
import pandas as pd

# k is the number of folds, model is the model we're testing
# the model is what we want to test, for hw2, it's the perceptron model we're using
# T is the number of epochs, passed into the model to train/test
def cross_validation(folds, k, model, T):
    total_accuracy = 0
    
    # train the model using the k-1 folds
    for i in range(k):
        # pick the testing fold
        test = folds[i]
        # pick the training folds, aka the folds that aren't the testing fold
        train = [fold for j, fold in enumerate(folds) if j != i]
        # train the model using the training folds
        model.train(train)
        # test the model using the testing fold
        accuracy = model.test(test)
        