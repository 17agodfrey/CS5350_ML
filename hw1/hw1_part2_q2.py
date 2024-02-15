from data import Data
import numpy as np
DATA_DIR = 'data/'

class Node: 
    def __init__(self, attribute):
        self.attribute = attribute
        self.branches = []
        self.label = None
        self.info_gain = None
        
    def add_branch(self, node, value):
        self.branches.append([node, value]) # value is the value of the attribute (branch) that leads to the node 
        
# S = set of examples
# attributes = set of of measured attributes        
def ID3(S, attributes, root = Node(None)):
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
        newNode = Node(None)
        root.add_branch(newNode, val)
        Sv = S.get_row_subset(A, val)
        Sv_len = Sv.__len__()
        if Sv.__len__() == 0:
            newNode.label = majority_label
        else:
            newNode = ID3(Sv, attributes, newNode)
    if root.label is None:
        root.label = majority_label
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
    entropy = -((num_e/total) * np.log2(num_e/total)) - ((num_p/total) * np.log2(num_p/total))
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
    # label_count_dict = col_label_counts(attribute)
    # if e == 0 or p == 0:
    #     return 0
    # else:
    e,p = label_counts(S.raw_data)
    ent = entropy(e,p)
    col_e = col_entropy(S, attribute)
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

    
    
# find tree with ID3 algorithm
# - start with S (all the rows/data basically) and attricutes (all the columns/attributes)

train = Data(fpath = DATA_DIR + 'train.csv')
test = Data(fpath = DATA_DIR + 'test.csv')

e,p = label_counts(train.raw_data)
label_entropy = entropy(e,p)
majority_label = 'e' if e > p else 'p'
decision_tree = ID3(train, train.attributes)

###DELETE LATER###
# print(train.get_attribute_possible_vals('cap-shape'))
# for val in train.get_attribute_possible_vals('cap-shape'):
#     print(f'{val}: {train.get_row_subset("cap-shape", val).__len__()}')
##################



####### DO NOT DELETE - USED FOR TESTING ########
# print(decision_tree.attribute)
# print(decision_tree.info_gain)
# print(tree_depth(decision_tree))
# print(test_accuracy(decision_tree, train))
# print(test_accuracy(decision_tree, test))
##################################################