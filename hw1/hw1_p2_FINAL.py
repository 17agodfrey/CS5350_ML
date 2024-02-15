import copy
from data import Data
import numpy as np
DATA_DIR = 'data/'

# from hw1_part2_q2 import train, test, majority_label
# from hw1_part2_q2 import bestAttribute, Node, test_accuracy, tree_depth 


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

def five_fold_cross_validation (folds, depth_limit):
    total_accuracy = 0
    total_std_dev = 0
    tree = None
    for i in range(0,5):
        test_fold = folds[i]
        train_folds_data = [fold.raw_data for j, fold in enumerate(folds) if j != i]
        train_folds_data = np.concatenate(train_folds_data, axis=0)

        train_folds = Data(data=train_folds_data)
        train_folds.attributes = copy.deepcopy(train.attributes)
        train_folds.index_column_dict = train.index_column_dict
        train_folds.column_index_dict = train.column_index_dict
                
        root = Node(None)
        tree = ID3_Depth_Limited(train_folds, train_folds.attributes, 0, depth_limit, root)
        total_accuracy += test_accuracy(tree, test_fold)
        
        
    print(f'    Average accuracy: {total_accuracy/5}')   
    std_dev = np.sqrt((total_accuracy/5)*(1-(total_accuracy/5))/(5))
    print(f'    Standard deviation: {std_dev}') 
    return tree

def ID3_Depth_Limited(S, attributes, current_depth, depth_limit, root = Node(None) ):
    # root = Node(None)
    # find attribute in attributes that best classifies S
    A, info_gain = bestAttribute(S, attributes)
    root.info_gain = info_gain
    if A is None:
        root.label = S.raw_data[0][0]
        return root
    root.attribute = A
    possible_vals = S.get_attribute_possible_vals(A)
    attributes.pop(A)
    
    for val in possible_vals:
        if current_depth < depth_limit:
            newNode = Node(None)
            root.add_branch(newNode, val)
            Sv = S.get_row_subset(A, val)
            Sv_len = Sv.__len__()
            if Sv.__len__() == 0:
                newNode.label = majority_label
            else:
                newNode = ID3_Depth_Limited(Sv, attributes, current_depth+1, depth_limit, newNode)
        else:
            root.label = majority_label
            break
    return root



# how many of each label for each possible value of an attribute
def col_label_counts(S, colName):
    e = 0
    p = 0
    possible_vals = S.get_attribute_possible_vals(colName)
    vals_dict = {}
    with_label = S.get_column(['label', colName])
    
    for val in possible_vals :
        vals_dict[val] = [0,0]
        
    for row in with_label :
        if row[0] == 'e' :
            vals_dict[row[1]][0] += 1
        else :
            vals_dict[row[1]][1] += 1
    
    return vals_dict
    

def entropy(num_e, num_p) :
    if num_e == 0 or num_p == 0:
        return 0  # or return a small non-zero value if you prefer
    total = num_e + num_p
    return -((num_e/total) * np.log2(num_e/total)) - ((num_p/total) * np.log2(num_p/total))

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
    e,p = label_counts(S.raw_data)
    return entropy(e,p) - col_entropy(S, attribute)
    
def bestAttribute(S, attributes):
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
    e = 0
    p = 0
    for row in array:
        if row[0] == 'e':
            e += 1
        else:
            p += 1
    return e,p

#################### NOT USED IN ID3 ####################

def tree_depth(node):
    if not node.branches:
        return 1  
    
    subtree_depths = [tree_depth(branch[0]) for branch in node.branches]
    return max(subtree_depths)+1

def test_accuracy(tree, data):
    correct = 0
    total = data.__len__()
    for row in data.raw_data:
        guess = classify(row, tree, data)
        actual = row[0]
        if guess == actual:
            correct += 1
    return correct / total

def classify(row, tree, data):
    if not tree.branches:
        return tree.label
    row_val = row[data.get_column_index(tree.attribute)]
    for branch in tree.branches: # the branches are the possible values of the current attribute (current attribute is the attribute of tree-which is a node)
        if branch[1] == row_val:
            return classify(row, branch[0], data) # recurse down the tree with the branch that matches the row's value
    return tree.label

#######################################################
    
# find tree with ID3 algorithm
# - start with S (all the rows/data basically) and attricutes (all the columns/attributes)

train = Data(fpath = DATA_DIR + 'train.csv')
test = Data(fpath = DATA_DIR + 'test.csv')

######### EXPERIMENT 1 #########

e,p = label_counts(train.raw_data)
label_entropy = entropy(e,p)
majority_label = 'e' if e > p else 'p'

decision_tree = ID3_Depth_Limited(train, train.attributes, 0, float('inf'))

# print('best feature:', decision_tree.attribute)
# print('root feature info gain: ', decision_tree.info_gain)
# print('max tree depth: ', tree_depth(decision_tree))
# print('accuracy on training set:', test_accuracy(decision_tree, train))
# print('accuracy on test set', test_accuracy(decision_tree, test))

### for submission: 
print('######## EXPERIMENT 1 ########')
print('(a) Entropy of the data: ', label_entropy)
print('(b) Best feature: ', decision_tree.attribute, ' info gain: ', decision_tree.info_gain)
print('(c) N/A')
print('(d) N/A')
print('(e) accuracy on training set:', test_accuracy(decision_tree, train))
print('(f) accuracy on test set', test_accuracy(decision_tree, test))


######### EXPERIMENT 2 #########

folds = []
# iterate through the 5 folds
for i in range(1,6):
    folds.append(Data(fpath = DATA_DIR + f'CVfolds_new/fold{i}.csv'))
    
# depths = [1, 2, 3, 4, 5, 10, 15]
# for depth in depths:
#     print(f'Cross-validation with depth limit: {depth} ----------')
#     five_fold_cross_validation(folds, depth)  

# best_tree = ID3_Depth_Limited(train, train.attributes, 0, 4)   
# best_tree_accuracy = test_accuracy(best_tree, test)
# print(f'best_tree_accuracy: ', best_tree_accuracy)

root = Node(None)
best_tree = ID3_Depth_Limited(train, train.attributes, 0, 4)
### for submission:
print('######## EXPERIMENT 2 ########')
print('(a) Entropy of the data: ', label_entropy)
print('(b) Best feature: ', best_tree.attribute, ' info gain: ', best_tree.info_gain)
print('(c) :')
train = Data(fpath = DATA_DIR + 'train.csv')
test = Data(fpath = DATA_DIR + 'test.csv')
depths = [1, 2, 3, 4, 5, 10, 15]
trees = []
for depth in depths:
    print(f'Cross-validation with depth limit: {depth} ----------')
    trees.append(five_fold_cross_validation(folds, depth))  
root = Node(None)
best_tree = trees[3]
print('(d) best tree depth: ', tree_depth(best_tree)-1)
print('(e) accuracy on training set:', test_accuracy(best_tree, train))
print('(f) accuracy on test set', test_accuracy(best_tree, test))


    
