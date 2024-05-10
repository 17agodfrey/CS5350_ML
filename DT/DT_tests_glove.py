import pandas as pd
import numpy as np
import DT_processing as dtp
import DT_helper as dth
import DT_ID3 as dtID3
import random
    
glove_train_df = pd.read_csv("project_data/data/glove/glove.train.csv")
glove_test_df = pd.read_csv("project_data/data/glove/glove.test.csv")
glove_eval_df = pd.read_csv("project_data/data/glove/glove.eval.anon.csv")

########### TOY DATA ###########
# test_data = {
#     'x1': [random.randint(0, 1) for _ in range(100)],
#     'x2': [random.randint(0, 1) for _ in range(100)],
# }

# labels = []
# for i in range(len(test_data['x1'])):
#     if test_data['x1'][i] == 1 and test_data['x2'][i] == 1: ## conjunction x1 ^ x2
#         labels.append(1)
#     else:
#         labels.append(0)
        
# test = pd.DataFrame(test_data)
# test.insert(0, 'label', labels)

# data = {
#     'x1': [0, 1, 0, 1],
#     'x2': [0, 0, 1, 1],
# }

# labels = []

# for i in range(len(data['x1'])):
#     if data['x1'][i] == 1 and data['x2'][i] == 1: ## conjunction x1 ^ x2
#         labels.append(1)
#     else:
#         labels.append(0)
        
# train = pd.DataFrame(data)
# train.insert(0, 'label', labels)
# glove_train_df = train
# attributes = glove_train_df.columns[1:].tolist()
# tree = dtID3.ID3_Depth_Limited(glove_train_df, glove_train_df.columns[1:].tolist(), 0, 5)
# accuracy = dth.test_accuracy(tree, test)
# print(accuracy)
################################



print('######## EXPERIMENT 1 : depth limiting ########')
depths = [3, 4, 5, 7, 10, 12, 15]
# depths = [5]

glove_train_labels = glove_train_df['label']
glove_train_discretized_attributes = dtp.discretize_columns(glove_train_df.drop(columns=['label']))
glove_train_df = pd.concat([glove_train_labels, glove_train_discretized_attributes], axis=1)

### find the best hyper parameter (depth) #### 
best_depth = 0
best_accuracy = 0
for depth in depths:
    accuracy = dtID3.k_fold_cross_validation(glove_train_df, 5, depth)
    print(f"Depth: {depth}, CV5 accuracy on training: {accuracy}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_depth = depth
print(f"Best depth: {best_depth}, best CV5 accuracy on training: {best_accuracy}")

## use the best hyper parameter to train the final tree ###
print("######## EXPERIMENT 2 : final tree ########")

try : 
    glove_train_labels = glove_train_df['label']
    glove_train_discretized_attributes = dtp.discretize_columns(glove_train_df.drop(columns=['label']))
    glove_train_df = pd.concat([glove_train_labels, glove_train_discretized_attributes], axis=1)
    S = glove_train_df
    print(S.head())
    attributes = glove_train_df.columns[1:].tolist()
    current_depth = 0
    depth_limit = 5
    tree = dtID3.ID3_Depth_Limited(S, attributes, current_depth, depth_limit)
    print(f"Depth: {depth_limit}, train Accuracy: {dth.test_accuracy(tree, glove_train_df)}")
    glove_test_labels = glove_test_df['label']
    glove_test_discretized_attributes = dtp.discretize_columns(glove_test_df.drop(columns=['label']))
    glove_test_df = pd.concat([glove_test_labels, glove_test_discretized_attributes], axis=1)
    print(f"Depth: {depth_limit}, test Accuracy: {dth.test_accuracy(tree, glove_test_df)}")
    glove_eval_labels = glove_eval_df['label']
    glove_eval_discretized_attributes = dtp.discretize_columns(glove_eval_df.drop(columns=['label']))
    glove_eval_df = pd.concat([glove_eval_labels, glove_eval_discretized_attributes], axis=1)
    ### we can't test accuracy on the eval data, as we don't have the labels for it. ###
    dth.eval_guesses_to_csv(tree, glove_eval_df, "glove.eval.guesses.csv")
    print("done")
except: 
    print("****error****")