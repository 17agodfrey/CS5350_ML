import numpy as np
from scipy.optimize import minimize

# Define the feature transformation function phi
def phi(x):
    return np.array([x[0]**2024, x[1]**2023])

# Positive and negative examples
positive_examples = [np.array([1, 1]), np.array([2, 2]), np.array([3, 3])]
negative_examples = [np.array([-1, -1]), np.array([-2, -2]), np.array([-3, -3])]

# Set up the optimization problem
def objective(w):
    b = w[2]
    return np.linalg.norm(w[:-1])**2

def constraint_positive(w):
    return w[:-1] @ phi(positive_examples[0]) - w[2]

def constraint_negative(w):
    return w[:-1] @ phi(negative_examples[0]) - w[2]

# Initial guess for w and b
w_initial = np.array([1, 1, 1])

# Solve the optimization problem
cons = [{'type': 'eq', 'fun': constraint_positive},
        {'type': 'eq', 'fun': constraint_negative}]
result = minimize(objective, w_initial, constraints=cons)

# Extract the optimal weight vector w and bias b
w_optimal = result.x[:-1]
b_optimal = result.x[-1]

print("Optimal weight vector:", w_optimal)
print("Optimal bias:", b_optimal)
