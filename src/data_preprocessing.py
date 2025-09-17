import os
import joblib
from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

class Preprocessing: 
    def __init__(self, data): 
        self.data = data.copy()
      
    def categorie(self, *column): 
        categories = lambda col : col.astype('category').cat.codes  
        self.data[list(column)] =  self.data[list(column)].apply(categories)
        
        return self
    
    def delete_column(self, *column): 
        for col in column: 
            self.data = self.data.drop(col, axis=1)
        
        return self
    
    def delete_ouliers(self, *column): 
        for col in column:
            Q1 ,Q3  = self.data[col].quantile([0.25,0.75]) 
            IQR = Q3 - Q1 
            lower, upper = Q1 - 1.5 * IQR , Q3 + 1.5 * IQR 
            # self.data.clip(lower=lower, upper=upper, axis=1)
            self.data = self.data.query(f"{col} >= @lower and {col} <=  @upper")
            
        return self
    
    def standarisation(self, *column): 
        scaler = StandardScaler()
        self.data[list(column)] = scaler.fit_transform(self.data[list(column)])
        
        return self
            
 

    def column_preprocessing(self, continuous, categorical, count_var, save_path="../preprocessor/preprocessor.pkl"):
        if not os.path.exists(save_path):
            os.makedirs("../preprocessor/", exist_ok=True)

        transformers = []
        
        if continuous:
            transformers.append(("num", StandardScaler(), continuous))
        
        if categorical:
            transformers.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical))
        
        if count_var:
            transformers.append(("count", "passthrough", count_var))

    
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")


        features = continuous + categorical + count_var
        preprocessor.fit(self.data[features])

       
        joblib.dump(preprocessor, save_path)
        print(f"Preprocessor : {save_path}")

        # return preprocessor

    def preprocessing(self, data_train_test, target): 
        
        X = data_train_test.drop(columns=[target], axis=1) 
        
        y = data_train_test[target]
        
        return X, y
    
    
    def get_data(self):
        return self.data