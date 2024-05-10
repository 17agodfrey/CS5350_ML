import pandas as pd
import numpy as np
import DT_helper as dth
import concurrent.futures



# glove_train_df = pd.read_csv("project_data/data/glove/glove.train.csv")
# glove_test_df = pd.read_csv("project_data/data/glove/glove.test.csv")
# glove_eval_df = pd.read_csv("project_data/data/glove/glove.eval.anon.csv")

# bow_train_df = pd.read_csv("project_data/data/bag-of-words/bow.train.csv")
# bow_test_df = pd.read_csv("project_data/data/bag-of-words/bow.test.csv")
# bow_eval_df = pd.read_csv("project_data/data/bag-of-words/bow.eval.anon.csv")

# tfidf_train_df = pd.read_csv("project_data/data/tfidf/tfidf.train.csv")
# tfidf_test_df = pd.read_csv("project_data/data/tfidf/tfidf.test.csv")
# tfidf_eval_df = pd.read_csv("project_data/data/tfidf/tfidf.eval.anon.csv")

# 5- fold cross validation: 
# < using ID3 but with a chosen limted depth > 
# 1. go through the 5 folds, and for each fold, use the other 4 to generate the tree with ID3
# 2. use each tree to test the accuracy of the fold that was left out.  
# 3. return the average accuracy of the 5 trees


class Node: 
    def __init__(self, attribute):
        self.attribute = attribute
        self.branches = []
        self.label = None
        self.info_gain = None
        
    def add_branch(self, node, value):
        self.branches.append([node, value]) # value is the value of the attribute (branch) that leads to the node 



##### SINGLE THREADING #####
def k_fold_cross_validation (df, num_folds, depth_limit,):
    total_accuracy = 0
    total_std_dev = 0
    tree = None
    
    folds = np.array_split(df, num_folds)
    
    for i in range(0,num_folds):
        test_fold = folds[i]
        train_folds = [fold for j, fold in enumerate(folds) if j != i]
        train_folds = pd.concat(train_folds, axis=0)
        root = Node(None)
        attributes = df.columns[1:].tolist()
        tree = ID3_Depth_Limited(train_folds, attributes, 0, depth_limit, root)
        total_accuracy += dth.test_accuracy(tree, test_fold)
        
    return total_accuracy / num_folds


###### MULTI-THREADING ######
# def k_fold_cross_validation(df, num_folds, depth_limit):
#     total_accuracy = 0
#     # tree = None
    
#     folds = np.array_split(df, num_folds)
    
#     def process_fold(train_folds, test_fold):
#         root = Node(None)
#         attributes = df.columns[1:].tolist()
#         tree = ID3_Depth_Limited(train_folds, attributes, 0, depth_limit, root)
#         return dth.test_accuracy(tree, test_fold)
    
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = []
#         for i in range(num_folds):
#             test_fold = folds[i]
#             train_folds = [fold for j, fold in enumerate(folds) if j != i]
#             train_folds = pd.concat(train_folds, axis=0)
#             futures.append(executor.submit(process_fold, train_folds, test_fold))
        
#         for future in concurrent.futures.as_completed(futures):
#             total_accuracy += future.result()

#     return total_accuracy / num_folds


def ID3_Depth_Limited(S, attributes, current_depth, depth_limit, root = Node(None) ):
    num_0 = S['label'].value_counts(dropna=False).get(0, 0)
    num_1 = S['label'].value_counts(dropna=False).get(1, 0)

    num_labels = num_0 + num_1
    if num_0 == num_labels or num_1 == num_labels:
        if num_0 == num_labels:
            root.label = 0
        else:
            root.label = 1
        return root    
    majority_label = determine_majority_label(S)
    if majority_label != 0 and majority_label != 1:
        print('hold on there partner')
    # find attribute in attributes that best classifies S
    best_A, best_A_info_gain = bestAttribute(S, attributes)
    root.info_gain = best_A_info_gain
    if best_A is None: ## this should only happen if all the attributes have been used, or the entropy is same on labels and attribute 
        # print('S: \n', S)  
        # if len(S) > 1:
        #     print('u effed up') 
        # print('S label: \n', S['label'].values[0])
        
        root.label = majority_label ##
        # root.label = S.raw_data[0][0]
        return root
    root.attribute = best_A
    possible_vals = S[best_A].unique()
    attributes.remove(best_A)
    
    for val in possible_vals:
        if current_depth < depth_limit:
            newNode = Node(None)
            root.add_branch(newNode, val)
            # print('S before:\n', S)
            Sv = S[S[best_A] == val][['label'] + attributes]
            if len(Sv) == 0:
                newNode.label = majority_label
            else:
                newNode = ID3_Depth_Limited(Sv, attributes, current_depth+1, depth_limit, newNode)
        else: ###NOTE: what're we doin here 
            root.label = majority_label
            break
    return root



# how many of each label for each possible value of an attribute
def col_label_counts(S, colName):
    e = 0
    p = 0
    possible_vals = S[colName].unique()
    vals_dict = {}
    # with_label = S.get_column(['label', colName])
    
    for val in possible_vals :
        vals_dict[val] = [0,0]
    
    for index, row in S.iterrows() :
        if row['label'] == 0 :
            vals_dict[row[colName]][0] += 1
        else :
            vals_dict[row[colName]][1] += 1
    
    return vals_dict 
    # dictionary of possible values of the attribute and the number of each label for each value
    # where first element of this list is num of 0's and second element is num of 1's

def entropy(num_0, num_1) :
    if num_0 == 0 or num_1 == 0:
        return 0  # or return a small non-zero value if you prefer
    total = num_0 + num_1
    return -((num_1/total) * np.log2(num_1/total)) - ((num_0/total) * np.log2(num_0/total))

# Information gain using entropy
def col_entropy(S, col):
    total_entropy = 0
    col_label_counts_dict = col_label_counts(S, col)
    total_num_rows = S.__len__()
    for val in col_label_counts_dict : 
        num_vals = col_label_counts_dict[val][0] + col_label_counts_dict[val][1]
        if num_vals != 0:
            total_entropy += entropy(col_label_counts_dict[val][0], col_label_counts_dict[val][1]) * (num_vals/total_num_rows)
    return total_entropy

def info_gain(S, attribute):
    # print(S['label'].value_counts())
    # print('S:\n', S)
    # print('attribute:\n', attribute)
    label_counts = S['label'].value_counts()
    zero = label_counts.get(0, 0) # get the count of the label 0, value 0 if None
    one = label_counts.get(1, 0)
    return entropy(zero,one) - col_entropy(S, attribute)
    
def bestAttribute(S, attributes):
    # print('S:\n', S)
    # print('attributes:\n', attributes)
    best = None
    best_gain = 0
    for a in attributes:
        gain = info_gain(S, a)
        if gain > best_gain:
            best_gain = gain
            best = a
    return best, best_gain

#return number of e and p in label column (first column)
def label_counts(array):
    zero = 0
    one = 0
    for row in array:
        if row[0] == 'e':
            zero += 1
        else:
            one += 1
    return zero, one

# def get_attributes(df):
#     return df.columns[1:]

def determine_majority_label(df):
    num_0 = df['label'].value_counts(dropna=False).get(0, 0)
    num_1 = df['label'].value_counts(dropna=False).get(1, 0)
    if num_0 > num_1:
        majority_label = 0
    else: 
        majority_label = 1

    return majority_label






