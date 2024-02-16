import numpy as np
import pandas as pd

# #### load dataset ####
# ## the dataset names are: diabetes.train.csv, diabetes.dev.csv
# ## and diabetes.test.csv.

# # load the dataset using pandas
# import pandas as pd
# train = pd.read_csv('hw2_data/diabetes.train.csv')
# dev = pd.read_csv('hw2_data/diabetes.dev.csv')
# test = pd.read_csv('hw2_data/diabetes.test.csv')

# ## CV split files: 
# train_0 = pd.read_csv('hw2_data/CVSplits/train0.csv')
# train_1 = pd.read_csv('hw2_data/CVSplits/train1.csv')
# train_2 = pd.read_csv('hw2_data/CVSplits/train2.csv')
# train_3 = pd.read_csv('hw2_data/CVSplits/train3.csv')
# train_4 = pd.read_csv('hw2_data/CVSplits/train4.csv')

#### some random notes on perceptron: 
# - the weights aren't reset after each epoch, they're continually updated throughout. 




##### 1. simple perceptron #####

## hyper parameter(s) ##

# num_T : number of epochs
# n : fixed learning rate, chosen from {1, 0.1, 0.01}

##----------------------------------------------

# w : weight vector - initialized to random num between -.01 and .01
# b : bias - initialized to random num between -.01 and .01

class SimplePerceptron :
    
        def __init__(self, train_folds, num_T, n):
            self.n = n
            self.w = np.random.uniform(-.01, .01, train_folds[0].shape[1]-1) # weight vector is same dimension as num of col of data 
            self.b = np.random.uniform(-.01, .01, 1)
            self.train_folds = train_folds
            self.T = 0
            self.num_T = num_T
            self.num_updates = 0

        
        # you may not even need data as an argument.
        # Remember x is a row of the data, y is the label
        def check(self, x, y):
            # print("x:\n", x)
            # print("w:\n", self.w)
            # print("b:\n", self.b)
            # print("y:\n", y)
            # print(y * (np.dot(self.w, x) + self.b))
            if (y * (np.dot(self.w, x) + self.b)) < 0:
                # print('incorrect')
                return False
            # print('correct')
            return True
        
        def update(self, x, y):
            # print("x:\n", x)
            # print("w:\n", self.w)
            self.w += self.n * y * x
            self.b += self.n * y
            self.num_updates += 1
            
            
        def test_accuracy(self, test_data):
            total_values = test_data.shape[0]
            correct_values = 0
            for i in range(test_data.shape[0]):
                x = test_data.iloc[i, 1:]
                y = test_data.iloc[i, 0]
                if self.check(x, y):
                    correct_values += 1
            # print(correct_values/total_values)
            return correct_values/total_values
        
        # basically this is training the model (updating w and b) over however many number of epochs we want
        def run_epoch(self, train_data):
            for i in range(train_data.shape[0]):
                x = train_data.iloc[i, 1:]
                y = train_data.iloc[i, 0]
                if not self.check(x, y):
                    self.update(x, y)
        
        def cross_validation(self):
            total_accuracy = 0
            k = len(self.train_folds) # number of folds
            for i in range(k): 
                # pick the testing fold
                test_data = self.train_folds[i]
                # print(test_data)
                # pick the training folds, aka the folds that aren't the testing fold
                train_data = pd.concat([fold for j, fold in enumerate(self.train_folds) if j != i], ignore_index=True)
                # print(train_data)
                # train the model using the training folds
                # print('----------new epoch set----------')
                for t in range(self.num_T):
                    # shuffle the data
                    train_data = train_data.sample(frac=1, random_state=42)
                    self.run_epoch(train_data)
                # test the model using the testing fold
                accuracy = self.test_accuracy(test_data)
                total_accuracy += accuracy
            return total_accuracy/k # average accuracy over k experiments
            
            

##### 2. Decaying learning rate #####

## hyper parameter(s) ##
# num_T : number of epochs

# n : decaying learning rate, initialized to {1, 0.1, 0.01} - decayed by n0/(1+t)
# t is the epoch number we are currently on 

class DecayingLearningRatePerceptron :
    
        def __init__(self, train_folds, num_T, n):
            self.n0 = n
            self.n = n
            self.w = np.random.uniform(-.01, .01, train_folds[0].shape[1]-1) # weight vector is same dimension as num of col of data 
            self.b = np.random.uniform(-.01, .01, 1)
            self.train_folds = train_folds
            self.T = 0
            self.num_T = num_T     
            self.num_updates = 0

               
        # you may not even need data as an argument.
        # Remember x is a row of the data, y is the label
        def check(self, x, y):
            if y * (np.dot(self.w, x) + self.b) <= 0:
                return False
            return True
        
        def update(self, x, y):
            self.w += self.n * y * x
            self.b += self.n * y
            self.num_updates += 1
        
        # call this when we've completed an epoch            
        def decay(self):
            self.n = self.n0/(1+self.T)
            self.T += 1
            
        def test_accuracy(self, test_data):
            total_values = test_data.shape[0]
            correct_values = 0
            for i in range(test_data.shape[0]):
                x = test_data.iloc[i, 1:]
                y = test_data.iloc[i, 0]
                if self.check(x, y):
                    correct_values += 1
            # print(correct_values/total_values)
            return correct_values/total_values
        
        # basically this is training the model (updating w and b) over however many number of epochs we want
        def run_epoch(self, train_data):
            for i in range(train_data.shape[0]):
                x = train_data.iloc[i, 1:]
                y = train_data.iloc[i, 0]
                if not self.check(x, y):
                    self.update(x, y)
            self.decay()
        
        def cross_validation(self):
            total_accuracy = 0
            k = len(self.train_folds) # number of folds
            for i in range(k): 
                # pick the testing fold
                test_data = self.train_folds[i]
                # pick the training folds, aka the folds that aren't the testing fold
                train_data = pd.concat([fold for j, fold in enumerate(self.train_folds) if j != i], ignore_index=True)
                # train the model using the training folds
                # print('----------new epoch set----------')
                for t in range(self.num_T):
                    # shuffle the data
                    train_data = train_data.sample(frac=1, random_state=42)
                    self.run_epoch(train_data)          
                self.T = 0
                # test the model using the testing fold
                accuracy = self.test_accuracy(test_data)
                total_accuracy += accuracy
            return total_accuracy/k # average accuracy over k experiments
            
            
##### 3. Margin Perceptron #####

## hyper parameter(s) ##
# num_T : number of epochs

# n : decaying learning rate, initialized to {1, 0.1, 0.01} - decayed by n0/(1+t)
# u : basically threshold for margin, initialized to {1, 0.1, 0.01}

## explanation: if y * (np.dot(self.w, x) + self.b) <= u, then update the weights
## so even if the prediction is 'correct' (correct sign), if the margin (prediction?) is less than u, update the weights

class MarginPerceptron : 
        
            def __init__(self, train_folds, num_T, n, u):
                self.n0 = n
                self.n = n
                self.u = u
                self.w = np.random.uniform(-.01, .01, train_folds[0].shape[1]-1) # weight vector is same dimension as num of col of data 
                self.b = np.random.uniform(-.01, .01, 1)
                self.train_folds = train_folds
                self.T = 0
                self.num_T = num_T    
                self.num_updates = 0

                     
            # you may not even need data as an argument.
            # Remember x is a row of the data, y is the label
            def check(self, x, y):
                if y * (np.dot(self.w, x) + self.b) <= self.u:
                    return False
                return True
            
            def update(self, x, y):
                self.w += self.n * y * x
                self.b += self.n * y
                self.num_updates += 1
            
            # call this when we've completed an epoch            
            def decay(self):
                self.n = self.n0/(1+self.T)
                self.T += 1
                
            def test_accuracy(self, test_data):
                total_values = test_data.shape[0]
                correct_values = 0
                for i in range(test_data.shape[0]):
                    x = test_data.iloc[i, 1:]
                    y = test_data.iloc[i, 0]
                    if self.check(x, y):
                        correct_values += 1
                # print(correct_values/total_values)
                return correct_values/total_values
        
            # basically this is training the model (updating w and b) over however many number of epochs we want
            def run_epoch(self, train_data):
                for i in range(train_data.shape[0]):
                    x = train_data.iloc[i, 1:]
                    y = train_data.iloc[i, 0]
                    if not self.check(x, y):
                        self.update(x, y)
                self.decay()
            
            def cross_validation(self):
                total_accuracy = 0
                k = len(self.train_folds) # number of folds
                for i in range(k): 
                    # pick the testing fold
                    test_data = self.train_folds[i]
                    # pick the training folds, aka the folds that aren't the testing fold
                    train_data = pd.concat([fold for j, fold in enumerate(self.train_folds) if j != i], ignore_index=True)
                    # print(train_data)
                    # train the model using the training folds
                    # print('----------new epoch set----------')
                    for t in range(self.num_T):
                        # shuffle the data
                        train_data = train_data.sample(frac=1, random_state=42)
                        self.run_epoch(train_data)          
                    self.T = 0                    
                    # test the model using the testing fold
                    accuracy = self.test_accuracy(test_data)
                    total_accuracy += accuracy
                return total_accuracy/k # average accuracy over k experiments
                    
                
##### 4. Averaged Perceptron #####

## hyper parameter(s) ##
# num_T : number of epochs

# n : fixed learning rate, initialized to {1, 0.1, 0.01}

## explanation: implement the averaged version of the orignal (1. simple perceptron) algorithm
## that is, keep track of the averaged weight vector 'a' and the averaged bias 'b' 
## nd update them at each iteration (epoch)

class AveragedPerceptron :
        
            def __init__(self, train_folds, num_T, n):
                self.n = n
                self.w = np.random.uniform(-.01, .01, train_folds[0].shape[1]-1) # weight vector is same dimension as num of col of data 
                self.b = np.random.uniform(-.01, .01, 1)
                self.a = np.zeros(train_folds[0].shape[1]-1) # averaged weight vector
                self.ba = 0 # averaged bias
                self.best_a = []
                self.best_ba = 0
                self.train_folds = train_folds
                self.T = 0
                self.num_T = num_T
                self.num_updates = 0
            
            # you may not even need data as an argument.
            # Remember x is a row of the data, y is the label
            def check(self, x, y):
                if y * (np.dot(self.a, x) + self.ba) <= 0:
                    return False
                return True
            
            def update(self, x, y):
                self.w += self.n * y * x
                self.b += self.n * y
                self.num_updates += 1
                
            def avg(self):
                self.a += self.w
                self.ba += self.b
                
            def test_accuracy(self, test_data):
                total_values = test_data.shape[0]
                correct_values = 0
                for i in range(test_data.shape[0]):
                    x = test_data.iloc[i, 1:]
                    y = test_data.iloc[i, 0]
                    if self.check(x, y):
                        correct_values += 1
                # print(correct_values/total_values)
                return correct_values/total_values
        
            # basically this is training the model (updating w and b) over however many number of epochs we want
            def run_epoch(self, train_data):
                for i in range(train_data.shape[0]):
                    x = train_data.iloc[i, 1:]
                    y = train_data.iloc[i, 0]
                    if not self.check(x, y):
                        self.update(x, y)
                    self.avg() # update the averaged weight vector and bias
            
            def cross_validation(self):
                total_accuracy = 0
                k = len(self.train_folds) # number of folds
                best_accuracy = -1
                for i in range(k): 
                    # pick the testing fold
                    test_data = self.train_folds[i]
                    # print(test_data)
                    # pick the training folds, aka the folds that aren't the testing fold
                    train_data = pd.concat([fold for j, fold in enumerate(self.train_folds) if j != i], ignore_index=True)
                    # print(train_data)
                    # train the model using the training folds
                    # print('----------new epoch set----------')
                    for t in range(self.num_T):
                        # shuffle the data
                        train_data = train_data.sample(frac=1, random_state=42)
                        self.run_epoch(train_data)          
                    accuracy = self.test_accuracy(test_data)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        self.best_a = self.a
                        self.best_ba = self.ba
                    total_accuracy += accuracy
                    self.a = np.zeros(train_data.shape[1]-1) # reset the averaged weight vector
                    self.ba = 0
                return total_accuracy/k # average accuracy over k experiments
            

            