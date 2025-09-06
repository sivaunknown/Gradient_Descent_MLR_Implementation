import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class MLR_Grad_Desc():
    
    def __init__(self,data : pd.DataFrame, lr : float) -> None:
        
        self.features = data.drop("income",axis=1).values
        self.actual_result = data["income"].values
        self.learn_rt = lr
        self.scalar = StandardScaler()
        self.mul_atts = self.scalar.fit_transform(self.features)
        self.mul_atts = np.c_[np.ones(self.mul_atts.shape[0]),self.mul_atts]
        self.n_samples,self.n_features = self.mul_atts.shape

    def fit(self,epochs=300) -> np.ndarray:
        
        wgt = np.zeros(self.n_features)
        for _ in range(epochs):
            wgt = self.grad_desc(wgt)
        self.wgt = wgt
        
    def grad_desc(self,wgt : np.ndarray) -> np.ndarray:
        
        pred_result = self.mul_atts.dot(wgt)
        error = pred_result - self.actual_result
        gradient = (2/self.n_samples) * self.mul_atts.T.dot(error)
        
        new_wgt = wgt - self.learn_rt * gradient
        return new_wgt
    
    def predict(self,new_mul_atts : pd.DataFrame) -> np.ndarray:
        
        new_mul_atts = self.scalar.transform(new_mul_atts)
        new_mul_atts = np.c_[np.ones(new_mul_atts.shape[0]),new_mul_atts]
        return self.new_mul_atts.dot(self.wgt)
    
 
data = pd.read_csv("Multiple Linear Regression/multiple_linear_regression_dataset.csv")
learn_rt = 0.01
mlr = MLR_Grad_Desc(data,0.01)
mlr.fit(epochs=300)
print(mlr.wgt)


'''
Note use predict method to test the model
'''
