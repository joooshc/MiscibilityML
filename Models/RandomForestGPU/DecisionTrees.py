import torch
import pandas as pd

# Defining the node data structure
def create_node(feature_index=None, threshold=None, value=None):
        return {
        'feature_index': feature_index,
        'threshold': threshold,
        'value': value,
        'left': None,
        'right': None
    }

# To check if a node is a leaf
def is_leaf(node):
    return node['value'] is not None

# The forward pass
def forward(node, x):
    if is_leaf(node):
        return torch.full((x.size(0),), node['value'], dtype=torch.float32, device=x.device)
    
    left_mask = x[:, node['feature_index']] < node['threshold']
    left_vals = forward(node['left'], x[left_mask])
    right_vals = forward(node['right'], x[~left_mask])

    output = torch.empty_like(x[:, 0], device=x.device)
    output[left_mask] = left_vals
    output[~left_mask] = right_vals

    return output

def build_tree(x, y, depth=3, max_depth=5):
    # If only one unique value, return leaf node
    if len(y.unique()) == 1:
        return create_node(value=y[0].item())

    # If maximum depth reached, return leaf with mean value
    if depth >= max_depth:
        return create_node(value=y.mean().item())
    
    best_split_feature = None
    best_split_threshold = None
    best_split_score = float('-inf')
    current_variance = y.var()

    # Find the best split
    for feature_idx in range(x.shape[1]):
        possible_thresholds = x[:, feature_idx].unique()
        for threshold in possible_thresholds:
            left_mask = x[:, feature_idx] < threshold
            right_mask = ~left_mask
            left_variance = y[left_mask].var() if left_mask.any() else 0
            right_variance = y[right_mask].var() if right_mask.any() else 0
            score = current_variance - (left_mask.float().mean() * left_variance + right_mask.float().mean() * right_variance)
            if score > best_split_score:
                best_split_score = score
                best_split_feature = feature_idx
                best_split_threshold = threshold.item()

    # If we didn't find a split, return leaf with mean value
    if best_split_feature is None:
        return create_node(value=y.mean().item())

    # Recursively build left and right subtrees
    left_mask = x[:, best_split_feature] < best_split_threshold
    right_mask = ~left_mask
    left_subtree = build_tree(x[left_mask], y[left_mask], depth=depth+1, max_depth=max_depth)
    right_subtree = build_tree(x[right_mask], y[right_mask], depth=depth+1, max_depth=max_depth)

    return create_node(feature_index=best_split_feature, threshold=best_split_threshold, left=left_subtree, right=right_subtree)       

# The data
'''
dataset_pt1 = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Log/MTrainLDm1.csv")
dataset_pt2 = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Log/MTrainLDm2.csv")
dataset_pt3 = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/Log/MTrainLDm3.csv")
y1 = dataset_pt1['MoleFraction']
y2 = dataset_pt2['MoleFraction']
y3 = dataset_pt3['MoleFraction']
X1 = dataset_pt1.iloc[:, 7:]
X2 = dataset_pt2.iloc[:, 7:]
X3 = dataset_pt3.iloc[:, 7:]

y_df = pd.concat([y1, y2, y3], axis=0).reset_index(drop=True)
X_df = pd.concat([X1, X2, X3], axis=0).reset_index(drop=True)
dataset = pd.concat([dataset_pt1, dataset_pt2, dataset_pt3], axis=0).reset_index(drop=True)
'''

# Without Dragon
dataset = pd.read_csv("C:/Users/dannt/Documents/GitHub/PSDI-miscibility2/Datasets/TrainTestData/WithoutDragon/MTrainQD.csv")
y_df = dataset['MoleFraction']
X_df = dataset.iloc[:, 7:]

# Convert dataframes to PyTorch tensors and move to GPU
X = torch.tensor(X_df.values, dtype=torch.float32).to('cuda')
y = torch.tensor(y_df.values, dtype=torch.float32).to('cuda')

root = build_tree(X, y, max_depth=3)
output = forward(root, X)
print(output)