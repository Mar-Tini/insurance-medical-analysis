from pathlib import Path 
import pandas as pd  

def load_data(path:  str)-> pd.DataFrame: 
    
    data_file =  Path(path)
    
    if not data_file.exists(): 
        raise FileExistsError(f'File not found {data_file}')
    
    if data_file.suffix == '.csv': 
        dataset = pd.read_csv(data_file)
        
        df = dataset.copy()
    
    else : 
        ValueError(f'file not supported')
        
    return df 