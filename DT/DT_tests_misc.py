import pandas as pd
import numpy as np
import DT_processing as dtp
import DT_helper as dth
import DT_ID3 as dtID3
import random
    
glove_train_df = pd.read_csv("project_data/data/glove/glove.train.csv")
glove_test_df = pd.read_csv("project_data/data/glove/glove.test.csv")
glove_eval_df = pd.read_csv("project_data/data/glove/glove.eval.anon.csv")

misc_train_df = pd.read_csv("project_data/data/misc/misc-attributes-train.csv")
misc_test_df = pd.read_csv("project_data/data/misc/misc-attributes-test.csv")
misc_eval_df = pd.read_csv("project_data/data/misc/misc-attributes-eval.csv")

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
depths = [1,2,3,4,5,6,7]
misc_train_df.insert(0, 'label', glove_train_df['label'])

### find the best hyper parameter (depth) #### 
best_depth = 0
best_accuracy = 0
for depth in depths:
    accuracy = dtID3.k_fold_cross_validation(misc_train_df, 5, depth)
    print(f"Depth: {depth}, CV5 accuracy on training: {accuracy}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_depth = depth
print(f"Best depth: {best_depth}, best CV5 accuracy on training: {best_accuracy}")


print("######## EXPERIMENT 2 : final tree ########")

try : 
    misc_train_df.insert(0, 'label', glove_train_df['label'])
    misc_test_df.insert(0, 'label', glove_test_df['label'])
    S = misc_train_df
    attributes = misc_train_df.columns[1:].tolist()
    current_depth = 0
    depth_limit = 5
    tree = dtID3.ID3_Depth_Limited(S, attributes, current_depth, depth_limit)
    print(f"Depth: {depth_limit}, train Accuracy: {dth.test_accuracy(tree, misc_train_df)}")
    print(f"Depth: {depth_limit}, test Accuracy: {dth.test_accuracy(tree, misc_test_df)}")
    ### we can't test accuracy on the eval data, as we don't have the labels for it. ###
    dth.eval_guesses_to_csv(tree, misc_eval_df, "misc.eval.guesses.csv")
    print("done")
except: 
    print("****error****")
