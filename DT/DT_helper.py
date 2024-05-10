import pandas as pd
import numpy as np
import DT_ID3 as dtID3

def tree_depth(node):
    if not node.branches:
        return 1  
    
    subtree_depths = [tree_depth(branch[0]) for branch in node.branches]
    return max(subtree_depths)+1

def test_accuracy(tree, data):
    correct = 0
    total = len(data)
    for row in data.iterrows():
        guess = classify(row, tree)
        actual = row[1]['label']
        if guess == actual:
            correct += 1
    return correct / total

def eval_guesses_to_csv(tree, data, filename):
    examples = []
    guesses = []
    i = 0
    # majority_label = dtID3.determine_majority_label(data)
    for row in data.iterrows():
        examples.append(i)
        guess = classify(row, tree) 
        if guess == None:
            guesses.append(1)
        else:
            guesses.append(classify(row, tree))
        i += 1
    examples_df = pd.DataFrame(examples, columns=['example_id'])
    guesses_df = pd.DataFrame(guesses, columns=['label'])
    guesses_df = pd.concat([examples_df, guesses_df], axis=1)
    guesses_df.to_csv(filename, index=False)

def classify(row, tree):
    if not tree.branches:
        return tree.label
    row_val = row[1][tree.attribute]
    for branch in tree.branches: # the branches are the possible values of the current attribute (current attribute is the attribute of tree-which is a node)
        if branch[1] == row_val:
            return classify(row, branch[0]) # recurse down the tree with the branch that matches the row's value
    return tree.label


# dummy_df = pd.DataFrame(np.random.randn(27, 6), columns=['A', 'B', 'C', 'D', 'E', 'F'])
# zeros_and_ones = np.random.randint(0, 2, size=27)[:, np.newaxis]
# dummy_df = pd.concat([pd.DataFrame(zeros_and_ones, columns=['label']), dummy_df], axis=1)

# print(dummy_df)
# for row in dummy_df.iterrows():
#     if 0 == row[1]['label']:
#         print(row[1]['label'])
