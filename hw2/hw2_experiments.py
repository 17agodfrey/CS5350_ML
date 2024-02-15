import hw2_perceptron as p
import numpy as np

#### load dataset ####
## the dataset names are: diabetes.train.csv, diabetes.dev.csv
## and diabetes.test.csv.

# load the dataset using pandas
import pandas as pd
train = pd.read_csv('hw2_data/diabetes.train.csv')
dev = pd.read_csv('hw2_data/diabetes.dev.csv')
test = pd.read_csv('hw2_data/diabetes.test.csv')

## CV split files: 
train_0 = pd.read_csv('hw2_data/CVSplits/train0.csv')
train_1 = pd.read_csv('hw2_data/CVSplits/train1.csv')
train_2 = pd.read_csv('hw2_data/CVSplits/train2.csv')
train_3 = pd.read_csv('hw2_data/CVSplits/train3.csv')
train_4 = pd.read_csv('hw2_data/CVSplits/train4.csv')

train_folds = [train_0, train_1, train_2, train_3, train_4]


##########TEST STUFF##########

### dot product test ###
# y=1
# w = np.random.uniform(-.01, .01, 4)
# x = train_folds[0].iloc[0, 1:5]
# b = .01
# print('w: ', w)
# print('x', x)


# print(y * (np.dot(w, x) + b))

### data shuffling test ###

# train_data = train_0.iloc[0:6, 0:6]
# print()
# for i in range(5):
#     train_data = train_data.sample(frac=1).reset_index(drop=True)
#     print(train_data)



##############################





##### 1. #####

# Run cross validation for ten epochs for each hyper-parameter combination to get the
# best hyper-parameter setting. Note that for cases when you are exploring combinations
# of hyper-parameters (such as the margin Perceptron), you need to try out all
# combinations.

# run cross validation for 10 epochs (10 epochs in each of the K experiments)
T = 10

### simple perceptron ###
# n_vals = [1, 0.1, 0.01]
# highest_accuracy = 0
# best_n = -1
# for n in n_vals:
#     model = p.SimplePerceptron(train_folds, T, n)
#     accuracy = model.cross_validation()
#     if accuracy > highest_accuracy:
#         highest_accuracy = accuracy
#         best_n = n
# print("Simple Perceptron")
# print("Best n: ", best_n, "\nAccuracy: ", highest_accuracy)


# ### decaying learning rate perceptron ###
# n_vals = [1, 0.1, 0.01]
# highest_accuracy = 0
# best_n = -1
# for n in n_vals:
#     model = p.DecayingLearningRatePerceptron(train_folds, T, n)
#     accuracy = model.cross_validation()
#     if accuracy > highest_accuracy:
#         highest_accuracy = accuracy
#         best_n = n
# print("Decaying Learning Rate Perceptron")
# print("Best n: ", best_n, "\nAccuracy: ", highest_accuracy)


# ### Margin Perceptron ###
# n_vals = [1, 0.1, 0.01]
# margin_vals = [1, 0.1, 0.01]
# highest_accuracy = 0
# best_combo = (-1, -1)   
# for n in n_vals:
#     for margin in margin_vals:
#         model = p.MarginPerceptron(train_folds, T, n, margin)
#         accuracy = model.cross_validation()
#         if accuracy > highest_accuracy:
#             highest_accuracy = accuracy
#             best_combo = (n, margin)
# print("Margin Perceptron")
# print("Best (n, margin): ", best_combo, "\nAccuracy: ", highest_accuracy)


### averaged perceptron ###
n_vals = [1, 0.1, 0.01]
highest_accuracy = 0
best_n = -1
best_sums = (0, 0)
for n in n_vals:
    model = p.AveragedPerceptron(train_folds, T, n)
    accuracy = model.cross_validation()
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        best_n = n
        best_sums = (model.best_a, model.best_ba)
print("Averaged Perceptron")
print("Best n: ", best_n, "\nAccuracy: ", highest_accuracy)
print("Best sums- \na: ", best_sums[0], '\nba: ', best_sums[1])


##### 2. #####
T= 20

### Simple Perceptron ###
model = p.SimplePerceptron(train_folds, T, 0.01)