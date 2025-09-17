import pandas as pd

def load_data(base_path='../data/processed/'):
    X_train = pd.read_csv(f'{base_path}X_train.csv')
    y_train = pd.read_csv(f'{base_path}y_train.csv')
    X_test  = pd.read_csv(f'{base_path}X_test.csv')
    y_test  = pd.read_csv(f'{base_path}y_test.csv')
    
    return X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
