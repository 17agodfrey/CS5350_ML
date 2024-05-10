#### README: This file is used to discretize the data, (continuous data to discrete data) 
#### this will likely be used for the glove and tfidf data, as the bag of words data is already discrete.


import pandas as pd
import numpy as np
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

glove_df = pd.read_csv("project_data/data/glove/glove.train.csv")
glove_labels_df = glove_df['label']
glove_data_df = glove_df.drop(columns=['label'])

# Display the first few rows of the DataFrame to verify it was read correctly
# print(glove_df.head())

glove_summary = glove_df.iloc[:, 1:].describe()

# print('glove summary:-------\n', glove_summary)


## calculate the number of bins for each column, using the Scott or Freedman-Diaconis rule
## only calculates how many bins there should be for each column, not the actual bin values
def calculate_bins(df, rule='scott'):
    num_samples = len(df)
    num_bins_dict = {}

    for col in df.columns:
        if col == 'label':
            continue
        if rule.lower() == 'scott':
            bin_width = 3.5 * df[col].std() / np.cbrt(num_samples)
        elif rule.lower() == 'freedman-diaconis':
            bin_width = 2 * (df[col].quantile(0.75) - df[col].quantile(0.25)) / np.cbrt(num_samples)
        else:
            raise ValueError("Invalid rule. Please choose 'scott' or 'freedman-diaconis'.")

        num_bins = (df[col].max() - df[col].min()) / bin_width
        num_bins = np.round(num_bins)
        num_bins_dict[col] = int(num_bins)
    
    return num_bins_dict


# this function discretizes the columns of a DataFrame
# that is, it converts the continuous values in each column to discrete values

# In other words::: the "cut" function calculates the new value of each element in the column 
# based on the number of bins in the column and assings it a value. 
## EXAMPLE: if the column has 5 bins, the cut function will assign each element in the column a value between 0 and 4
## lower values are assigned to lower bins, higher values are assigned to higher bins
def discretize_columns(df):
    num_bins_dict = calculate_bins(df)
    discretized_df = pd.DataFrame()
    
    for col, num_bins in num_bins_dict.items():
        discretized_df[col] = pd.cut(df[col], bins=10, labels=False)
    
    return discretized_df


# bins = calculate_bins(glove_data_df)
# for _bin in bins:
#     print(f"{_bin}: {bins[_bin]}")


# discretized_df = discretize_columns(glove_df.iloc[:, 1:])
# print('glove_df head-----------\n',glove_data_df.head())
# print('discretized_df head-----------\n',discretized_df.head())
# print('discredized_df tail-----------\n',discretized_df.tail())


### QUESTION: say we have a column with 5 bins, and a column with 10 bins
### what happens if, for the same row, both columns have the same value?
### will the discretized values be the same?

### MY ANSWER: we don't want to comapre the values of columns anyways
### because the columns are different, so the values are different

### (copilot) ANSWER: yes, the discretized values will be the same, because the cut function
# write code to prove this: 
# Create a DataFrame with 10 rows
# data = {
#     'label': np.random.randint(0, 2, size=100),  # Generate random 0s and 1s for the label column
#     'col1': np.random.randint(0, 100, size=100),  # Generate random integers for col1
# }

# # Ensure col2 has twice the range of col1
# range_col1 = data['col1'].max() - data['col1'].min()
# data['col2'] = np.random.randint(data['col1'].min(), data['col1'].max() + range_col1, size=100)

# df = pd.DataFrame(data)
# print(df)
# discretized_df = discretize_columns(df)
# print("Discretized DataFrame:\n", discretized_df)

