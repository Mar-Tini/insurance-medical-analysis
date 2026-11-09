from src.data.load import load_data 
from src.data.config import load_config
from src.data.preprocessing import preprocessing_data 

def run_pipeline():
    
    data_config, model_config = load_config()
    
    df = load_data(data_config['data']['raw'])
    
    preprocessing_data(df)
    
if __name__ == '__main__': 
    run_pipeline()