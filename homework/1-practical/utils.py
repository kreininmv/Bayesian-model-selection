import pandas as pd
import seaborn as sns

def get_X_y(num=1, to_numpy=False):
    X = pd.read_csv(f'./data/task1_{num}_learn_X.csv', header=None, sep=' ')
    y = pd.read_csv(f'./data/task1_{num}_learn_y.csv', header=None, sep=' ')
    if to_numpy:
        return X.to_numpy(), y.to_numpy()
    return X, y

def get_test(num=1, to_numpy=False):
    X = pd.read_csv(f'./data/task1_{num}_test_X.csv', header=None, sep=' ')
    if to_numpy:
        return X.to_numpy()
    return X

def draw_stats(X, y):
    #print(X.describe())
    sns.heatmap(X.corr())
    data = X.copy()
    data['answers'] = y
    sns.pairplot(data=data, hue='answers')