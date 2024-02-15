import copy
from data import Data
import numpy as np
DATA_DIR = 'data/'

from hw1_part2_q2 import train, test, majority_label
from hw1_part2_q2 import bestAttribute, Node, test_accuracy, tree_depth 


# 5- fold cross validation: 
# < using ID3 but with a chosen limted depth > 
# 1. go through the 5 folds, and for each fold, use the other 4 to generate the tree with ID3
# 2. use each tree to test the accuracy of the fold that was left out.  
# 3. return the average accuracy of the 5 trees

def five_fold_cross_validation (folds, depth_limit):
    total_accuracy = 0
    for i in range(0,5):
        test_fold = folds[i]
        # train_folds = [fold.raw_data for j, fold in enumerate(folds) if j != i]
        # Perform element-wise addition along the columns (axis=0)
        
        # train_folds = np.concatenate(train_folds, axis=0)
        # train_folds = Data(data = train_folds)
        # train_folds.attributes = copy.deepcopy(train.attributes)
        # train_folds.index_column_dict = train.index_column_dict
        # train_folds.column_index_dict = train.column_index_dict
        
        tree = Node(None)
        for j in range(5):
            if j != i:
                fold = folds[j]
                print(fold.raw_data.__len__())
                # root = Node(None)
                tree = ID3_Depth_Limited(fold, fold.attributes, 0, depth_limit, tree)
                print(f'Tree depth: {tree_depth(tree)-1}')
        total_accuracy += test_accuracy(tree, test_fold)


        
        
        
        # root = Node(None)
        # tree = ID3_Depth_Limited(train_folds, train_folds.attributes, 0, depth_limit, root)

        # print(f'train_folds length: {train_folds.__len__()}')
        # total_a = test_accuracy(tree, test_fold)
    print(f'Average accuracy: {total_accuracy/5}')    
    

def ID3_Depth_Limited(S, attributes, current_depth, depth_limit, root = Node(None) ):
    # root = Node(None)
    # find attribute in attributes that best classifies S
    A, info_gain = bestAttribute(S, attributes)
    if root.info_gain is not None and info_gain < root.info_gain:
        return root

    root.info_gain = info_gain
    if A is None:
        root.label = majority_label
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
    return root


train = Data(fpath = DATA_DIR + 'train.csv')

folds = []
# iterate through the 5 folds
for i in range(1,6):
    folds.append(Data(fpath = DATA_DIR + f'CVfolds_new/fold{i}.csv'))

depths = [1, 2, 3, 4, 5, 10, 15]
for depth in depths:
    print(f'Cross-validation with depth limit: {depth} ----------')
    five_fold_cross_validation(folds, depth)  
    
