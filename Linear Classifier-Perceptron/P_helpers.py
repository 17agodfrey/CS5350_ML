import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import Perceptron as p


# import numpy as np
# import pandas as pd
# import Perceptron as p
# import pygame



########## HELPER FUNCTIONS ##########
def eval_guesses_to_csv(a, ba, data, filename):
    examples = []
    guesses = []
    i = 0
    # majority_label = dtID3.determine_majority_label(data)
    for row in data.iterrows():
        examples.append(i)
        # print(row[1].shape)
        # print(a.shape)
        # print(a)
        guessLabel = guess(row[1].values, a, ba)
        if guessLabel == None:
            guesses.append(1)
        else:
            guesses.append(guessLabel)
        i += 1
    examples_df = pd.DataFrame(examples, columns=['example_id'])
    guesses_df = pd.DataFrame(guesses, columns=['label'])
    guesses_df = pd.concat([examples_df, guesses_df], axis=1)
    guesses_df.to_csv(filename, index=False)


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
    
def guess(x, w, b):
    # print("x:\n", x)
    # print("w:\n", self.w)
    # print("b:\n", self.b)
    # print("y:\n", y)
    # print(y * (np.dot(self.w, x) + self.b))
    print(x)
    print(np.dot(w, x) + b)
    if (np.dot(w, x) + b) < 0:
        # print('incorrect')
        return 0
    # print('correct')
    return 1    



# BOW_eval = pd.read_csv('../project_data/data/bag-of-words/bow.eval.anon.csv').drop(columns=['label'])
# a = pd.read_csv('a.csv', header=None).to_numpy().flatten()
# ba = pd.read_csv('ba.csv', header=None).to_numpy()[0][0]

# eval_guesses_to_csv(a, ba, BOW_eval, "glove.eval.guesses.csv")
    
