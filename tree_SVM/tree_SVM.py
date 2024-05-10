import DT_ID3 as dtID3
import DT_helper as dth
import SVM as svm
import pandas as pd
import numpy as np

glove_train_df = pd.read_csv("project_data/data/glove/glove.train.csv")
glove_test_df = pd.read_csv("project_data/data/glove/glove.test.csv")
glove_eval_df = pd.read_csv("project_data/data/glove/glove.eval.anon.csv")

misc_train_df = pd.read_csv("project_data/data/misc/misc-attributes-train.csv")
misc_test_df = pd.read_csv("project_data/data/misc/misc-attributes-test.csv")
misc_eval_df = pd.read_csv("project_data/data/misc/misc-attributes-eval.csv")


depth_limit = 5
misc_train_df.insert(0, 'label', glove_train_df['label'])
misc_test_df.insert(0, 'label', glove_test_df['label'])

##train the 100 trees 
DT_set = []
for i in range(100):
    train_sample = misc_train_df.sample(n=1000)
    train_sample_attributes = train_sample.columns[1:].tolist()    
    DT = dtID3.ID3_Depth_Limited(train_sample, train_sample_attributes, 0, depth_limit)
    DT_set.append(DT)
    print(f"Tree {i} trained")

train_transformed = np.zeros((len(misc_train_df), 100)) # 100 featues, same # rows 
train_data = misc_train_df.drop('label', axis=1)
print("train_data shape: ", train_data.shape)
for i in range(len(misc_train_df)):
    for j in range(100):
        try :
            train_transformed[i][j] = dth.classify(misc_train_df.iloc[i], DT_set[j])
        except Exception as e:
            print(e)
            print(i,j)

train_transformed = pd.DataFrame(train_transformed)
train_transformed.insert(0, 'label', misc_train_df['label'])            


## determine best hyperparameters for SVM using transformed data
# LR_0_arr = [1, .1, .01, .001, .0001, .00001]
# C_arr = [1, .1, .01, .001, .0001, .00001]
T = 100 # we probs (shouldn't) reach this, cause it should stop on its own cause we'll converge to the bottom of the bowl
SVM_obj_threshold = 0.001
# bestHyperparameters = {"LR_0": 0, "C": 0, "AvgAccuracy": 0}

# train_folds = np.array_split(train_transformed,5)

# for LR_0 in LR_0_arr:
#     for C in C_arr:
#         AvgAccuracy = svm.CV(train_folds, C, LR_0, T, SVM_obj_threshold)
#         print(f'LR_0 = {LR_0}, C = {C}, AvgAccuracy = {AvgAccuracy}')
#         if AvgAccuracy > bestHyperparameters["AvgAccuracy"]:
#             bestHyperparameters["LR_0"] = LR_0
#             bestHyperparameters["C"] = C
#             bestHyperparameters["AvgAccuracy"] = AvgAccuracy
# print("best Hyperparameters: ", bestHyperparameters)          


## make new dataset - get predictions from each tree for each test example 
test_transformed = np.zeros((len(misc_test_df), 100))
test_data = misc_test_df.drop('label', axis=1)
for i in range(len(test_transformed)):
    for j in range(100):
        try: 
            test_transformed[i][j] = dth.classify(test_data.iloc[i], DT_set[j])   
        except: 
            print(i, j)
test_transformed = pd.DataFrame(test_transformed)
test_transformed.insert(0, 'label', misc_test_df['label'])


## finally, test on transformed test data
train_data = train_transformed.drop('label', axis=1).values
train_labels = train_transformed['label'].replace(0, -1).values
test_data = test_transformed.drop('label', axis=1).values
test_labels = test_transformed['label'].replace(0, -1).values

LR_0 = 1
C = 1
w = svm.stochastic_sub_gradient_descent_SVM(train_data, train_labels, LR_0, C, T, SVM_obj_threshold)

N = len(test_data)
correct = 0
accuracy = 0
predict_0 = 0
predict_1 = 0
for i in range(N):
    # this is where we need to do the false/true positive/negative stuff
    prediction = np.dot(w, test_data[i])
    actual = test_labels[i] 
    if prediction > 0: predict_1 += 1
    else: 
        predict_0 += 1
        precition = -1
        print("prediction: ", prediction)

    if prediction > 0 and actual == 1:
        correct += 1
    elif prediction <= 0 and actual == -1:
        correct += 1
accuracy = correct/N


print("On Test Set: -------------------")
print("N", N)
print("predict_0", predict_0)
print("predict_1", predict_1)
print("Accuracy: ", accuracy)


## transform eval data
eval_transformed = np.zeros((len(misc_eval_df), 100))
eval_data = misc_eval_df
for i in range(len(eval_transformed)):
    for j in range(100):
        try: 
            eval_transformed[i][j] = dth.classify(eval_data.iloc[i], DT_set[j])
        except:
            print(i, j)
eval_transformed = pd.DataFrame(eval_transformed)
svm.eval_guesses_to_csv(w, eval_transformed, "tree_SVM.misc.eval.predictions.csv")
