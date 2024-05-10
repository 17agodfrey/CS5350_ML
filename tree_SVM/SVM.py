import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def stochastic_sub_gradient_descent_SVM(train_data, train_labels, LR_0, C, T, SVM_obj_threshold) :
    N = len(train_data)
    w = np.zeros(len(train_data[0]))
    LR_t = LR_0
    prev_SVM_obj = None
    for t in range(T):
        # shuffle the data
        indices = list(range(N))
        random.shuffle(indices) ### this may not work, 
        # calculate the learning rate
        LR_t = LR_0/(1+t)
        for i in range(N):
            yi = train_labels[i]
            xi = train_data[i]
            if yi * np.dot(w, xi) <= 1:
                w = (1-LR_t)*w + LR_t*C*(yi * xi)
            else:
                w = (1-LR_t)*w
        # calculate SVM objective after each epoch, may have to sum but idk
        SVM_obj = np.sum(np.maximum(0, 1 - train_labels * np.dot(train_data, w))) + 0.5 * C * np.dot(w, w)
        # may not even need this part
        if prev_SVM_obj == None:
            prev_SVM_obj = SVM_obj
            continue
        # stop iterating if change in SVM objective is smaller than threshold
        if prev_SVM_obj - SVM_obj < SVM_obj_threshold:
            break
        prev_SVM_obj = SVM_obj
    return w




## modifying this from assignment to use accuracy instead of F_1

def CV (train_folds, C, LR_0, T, SVM_obj_threshold):
    accuracy = 0
    for i in range(5):
        test_data = train_folds[i].drop('label', axis=1).values
        test_labels = train_folds[i]['label'].values
        train_data = pd.concat([train_folds[j] for j in range(5) if j != i]).drop('label', axis=1).values
        train_labels = pd.concat([train_folds[j] for j in range(5) if j != i])['label'].values
        w = stochastic_sub_gradient_descent_SVM(train_data, train_labels, LR_0, C, T, SVM_obj_threshold)
        N = len(test_data)
        correct = 0
        for i in range(N):
            # this is where we need to do the false/true positive/negative stuff
            prediction = np.dot(w, test_data[i])
            actual = test_labels[i] 
            if prediction > 0 and actual == 1:
                correct += 1
            elif prediction < 0 and actual == -1:
                correct += 1
        accuracy += correct/N
            
    return accuracy/5


def eval_guesses_to_csv(w, data, filename):
    examples = []
    guesses = []
    i = 0
    # majority_label = dtID3.determine_majority_label(data)
    for i in range(len(data)):
        examples.append(i)
        # print(row[1].shape)
        # print(a.shape)
        # print(a)
        guessLabel = np.dot(w, data.iloc[i])
        if guessLabel > 0:
            guesses.append(1)
        else:
            guesses.append(0)
        i += 1
    examples_df = pd.DataFrame(examples, columns=['example_id'])
    guesses_df = pd.DataFrame(guesses, columns=['label'])
    guesses_df = pd.concat([examples_df, guesses_df], axis=1)
    guesses_df.to_csv(filename, index=False)