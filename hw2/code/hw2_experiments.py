import hw2_perceptron as p
import numpy as np
import matplotlib.pyplot as plt


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


########## HELPER FUNCTIONS ##########
def test_accuracy_classifiers(test_data, w, b):
    total_values = test_data.shape[0]
    correct_values = 0
    for i in range(test_data.shape[0]):
        x = test_data.iloc[i, 1:]
        y = test_data.iloc[i, 0]
        if check(x,y,w,b):
            correct_values += 1
    # print(correct_values/total_values)
    return correct_values/total_values

def check(x, y, w, b):
    # print("x:\n", x)
    # print("w:\n", self.w)
    # print("b:\n", self.b)
    # print("y:\n", y)
    # print(y * (np.dot(self.w, x) + self.b))
    if (y * (np.dot(w, x) + b)) < 0:
        # print('incorrect')
        return False
    # print('correct')
    return True

def plot_learning_curve(acc_dict, title):
    plt.figure()
    x = list(acc_dict.keys())
    y = list(acc_dict.values())
    plt.plot(x, y)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.grid(True)
    plt.savefig(title + '_LearningCurve' + '.png')


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


# yeet = (1, 2)
# print(yeet[0])
# print(yeet[1])


##############################





print('######## 4.3 ########\n\n\n')

# Run cross validation for ten epochs for each hyper-parameter combination to get the
# best hyper-parameter setting. Note that for cases when you are exploring combinations
# of hyper-parameters (such as the margin Perceptron), you need to try out all
# combinations.

print('#### 4.3.1 - 10 epochs ####\n')
# run cross validation for 10 epochs (10 epochs in each of the K experiments)
T = 10

print('## Simple Perceptron ##')
n_vals = [1, 0.1, 0.01]
highest_accuracy = 0
best_n = -1
for n in n_vals:
    model = p.SimplePerceptron(train_folds, T, n)
    accuracy = model.cross_validation()
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        best_n = n
print("Best n: ", best_n, "\nAccuracy: ", highest_accuracy)


print('\n## Decaying Learning Rate Perceptron ##')
n_vals = [1, 0.1, 0.01]
highest_accuracy = 0
best_n = -1
for n in n_vals:
    model = p.DecayingLearningRatePerceptron(train_folds, T, n)
    accuracy = model.cross_validation()
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        best_n = n
print("Best n: ", best_n, "\nAccuracy: ", highest_accuracy)


print('\n## Margin Perceptron ##')
n_vals = [1, 0.1, 0.01]
margin_vals = [1, 0.1, 0.01]
highest_accuracy = 0
best_combo = (-1, -1)   
for n in n_vals:
    for margin in margin_vals:
        model = p.MarginPerceptron(train_folds, T, n, margin)
        accuracy = model.cross_validation()
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_combo = (n, margin)
print("Best (n, margin): ", best_combo, "\nAccuracy: ", highest_accuracy)


print('\n## Averaged Perceptron ##')
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
print("Best n: ", best_n, "\nAccuracy: ", highest_accuracy)
print("Best sums- \na: ", best_sums[0], '\nba: ', best_sums[1])


print('\n\n#### 4.3.2 & .3 - 20 epochs ####\n')
#(using best hyper-parameters from 1)
T= 20

### Simple Perceptron ###
model = p.SimplePerceptron(train_folds, T, 0.1)
best_accuracy = -1
best_classifiers = ([], -1)
acc_dict = {}
for t in range(T):
    model.run_epoch(train)
    accuracy = model.test_accuracy(dev)
    acc_dict[t] = accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_classifiers = (model.w, model.b)
        
print("## Simple Perceptron ##")
print("Total number of updates performed on training set: ", model.num_updates)
print("Best accuracy on dev set within 20 epochs: ", best_accuracy)
w = best_classifiers[0]
b = best_classifiers[1]
print("Accuracy on test set: ", test_accuracy_classifiers(test, w, b))
plot_learning_curve(acc_dict, 'Simple Perceptron')
    
    
### Decaying Learning Rate Perceptron ###
model = p.DecayingLearningRatePerceptron(train_folds, T, .01)
best_accuracy = -1
best_classifiers = ([], -1)
acc_dict = {}
for t in range(T):
    train = train.sample(frac=1, random_state=42)
    model.run_epoch(train)
    accuracy = model.test_accuracy(dev)
    acc_dict[t] = accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_classifiers = (model.w, model.b)
        
print("\n## Decaying Learning Rate Perceptron ##")
print("Total number of updates performed on training set: ", model.num_updates)
print("Best accuracy on dev set within 20 epochs: ", best_accuracy)
w = best_classifiers[0]
b = best_classifiers[1]
print("Accuracy on test set: ", test_accuracy_classifiers(test, w, b))
plot_learning_curve(acc_dict, 'Decaying Learning Rate Perceptron')


### Margin Perceptron ###
model = p.MarginPerceptron(train_folds, T, .01, .01)
best_accuracy = -1
best_classifiers = ([], -1)
acc_dict = {}
for t in range(T):
    train = train.sample(frac=1, random_state=42)
    model.run_epoch(train)
    accuracy = model.test_accuracy(dev)
    acc_dict[t] = accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_classifiers = (model.w, model.b)
        
print("\n## Margin Perceptron ##")
print("Total number of updates performed on training set: ", model.num_updates)
print("Best accuracy on dev set within 20 epochs: ", best_accuracy)
w = best_classifiers[0]
b = best_classifiers[1]
print("Accuracy on test set: ", test_accuracy_classifiers(test, w, b))
plot_learning_curve(acc_dict, 'Margin Perceptron')



### Averaged Perceptron ###
model = p.AveragedPerceptron(train_folds, T, 0.1)
best_accuracy = -1
sum_classifiers = ([], -1)
acc_dict = {}
for t in range(T):
    train = train.sample(frac=1, random_state=42)
    model.run_epoch(train)
    accuracy = model.test_accuracy(dev)
    acc_dict[t] = accuracy
sum_classifiers = (model.a, model.ba)
        
print("\n## Averaged Perceptron ##")
print("Total number of updates performed on training set: ", model.num_updates)
print("Best accuracy on dev set after 20 epochs: ", test_accuracy_classifiers(dev, sum_classifiers[0], sum_classifiers[1]))
# print("Best accuracy on dev set within 20 epochs: ", best_accuracy)
w = sum_classifiers[0]
b = sum_classifiers[1]
print("Accuracy on test set: ", test_accuracy_classifiers(test, w, b))
plot_learning_curve(acc_dict, 'Averaged Perceptron')


print('\n\n\n######## 4.4 ########')

print('\n### 2. ###') 
majority_label_count = dev.iloc[:, 0].value_counts().max()
total_labels = len(dev.iloc[:, 0])
majority_label_percentage = (majority_label_count / total_labels) * 100
print("Dev Majority label percentage: ", majority_label_percentage)

majority_label_count = test.iloc[:, 0].value_counts().max()
total_labels = len(test.iloc[:, 0])
majority_label_percentage = (majority_label_count / total_labels) * 100
print("Test Majority label percentage: ", majority_label_percentage)



    
    
    


