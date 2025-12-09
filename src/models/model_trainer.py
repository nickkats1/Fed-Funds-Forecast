from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error,r2_score,mean_absolute_percentage_error
from xgboost import XGBRegressor

from src.datasets.data_ingestion import DataIngestion
from src.datasets.data_transformation import DataTransformation

from tools.config import load_config
from tools.logger import logger






class ModelTrainer:
    """Train models for MLFlow and hyper-parameter tuning."""
    
    def __init__(self,config: dict, data_ingestion: DataIngestion | None = None):
        """
        Initializing ModelTrainer class.
        
        Args:
            config (dict): Configuration file.
            data_transformation (DataTransformation):  A instance of the DataTransformation class.
        """
        
        self.config = config or load_config()
        self.data_ingestion = data_ingestion or DataIngestion(self.config)  
        
        
        
    def load_models(self):
        """
        params and models loaded for GridSearchCV.
        
        Returns:
            models: sklearn model's for training.
            params: parameters for hyperparameter tuning.
        """
        
        params = {
            "LinearRegression_params": {
                "fit_intercept":[True],
                "copy_X": [True,False],
                "n_jobs": [1000,1500,2000],
                "positive": [True,False]
            },
            "Lasso_params": {
                "alpha": [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]
            },
            "Ridge_params": {
                "alpha": [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]
            },
            "GradientBoostingRegressor_params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 4, 5],
                "min_samples_split": [1, 5,10]
            },
            "RandomForestRegressor_params": {
                "n_estimators": [50, 100, 200],
                "min_samples_leaf": [1,2,4],
                "max_features": ['sqrt', 'log2', None]
            },
            "BaggingRegressor_params": {
                "n_estimators": [50,100,200],
                "max_samples": [1.0,0.8,0.6],
                "max_features": [1.0,0.8,0.6]
            },
            "XGBRegressor_params": {
                "n_estimators": [100,200,300],
                "max_depth": [3,5,9],
                "min_child_weight": [1,3,5],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.6, 0.8],
                "colsample_bytree": [0.6, 0.8]
            },
            "DecisionTreeRegressor_params": {
                "max_depth": [None,10,15],
                "min_samples_split": [2,5,10],
                "min_samples_leaf": [1,2,5]
            },
        }
        
        # models with hyper-parameters
        
        models = {
            "LinearRegression":(LinearRegression(),params["LinearRegression_params"]),
            "Lasso":(Lasso(),params["Lasso_params"]),
            "Ridge": (Ridge(), params["Ridge_params"]),
            "GradientBoostingRegressor":(GradientBoostingRegressor(),params["GradientBoostingRegressor_params"]),
            "RandomForestRegressor": (RandomForestRegressor(),params["RandomForestRegressor_params"]),
            "BaggingRegressor":(BaggingRegressor(),params["BaggingRegressor_params"]),
            "XGBRegressor": (XGBRegressor(),params["XGBRegressor_params"]),
            "DecisionTreeRegressor":(DecisionTreeRegressor(),params["DecisionTreeRegressor_params"])
        }
        return params,models
    
    def split(self):
        """Split data for training and testing"""
        # get data through DataIngestion
        
        data = self.data_ingestion.fetch_fred_data()
        
        # features and targets
        
        X = data.drop("FEDFUNDS",axis=1)
        y = data["FEDFUNDS"]
        
        # train/test split
        
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=1)
        
        
        # MinMax Scaler
        scaler = MinMaxScaler()
        
        
        # scaled training and testing data
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled,X_test_scaled,y_train,y_test
        
    
    def train(self):
        """Train models through GridSearchCV"""
        params,models = self.load_models()
        
        # get X_train_scaled,X_test_scaled,y_train,y_test
        
        X_train_scaled,X_test_scaled,y_train,y_test = self.split()
        
        
        
        
        for model_name,(model,params) in models.items():
            
            
            # grid-searcg
            grid_search = GridSearchCV(model,params,cv=4,scoring="neg_mean_squared_error",n_jobs=-1)
            # fit grid search
            grid_search.fit(X_train_scaled,y_train)
            
            y_pred = grid_search.predict(X_test_scaled)
            
            r2 = r2_score(y_test,y_pred)
            print(f"R2 Score->Model Name: {model_name} {r2*100:.2f}")
            
            mape = mean_absolute_percentage_error(y_test,y_pred)
            print(f"Mean-Absolute Percentage Error: {model_name}--{mape:.4}")
            # best params
            
            best_params = grid_search.best_params_
            
            # best score
            
            best_score = grid_search.best_score_
            
            print(f"Best Score for model: {model_name}<===>{best_score}")
            
            print(f"Best Params for model: {model_name}<====>{best_params}")