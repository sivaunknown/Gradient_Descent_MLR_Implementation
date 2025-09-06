import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class MLR_Stud_Performance():

    def __init__(self,data : pd.DataFrame, lr : float) -> None:

        self.data = data
        self.features = self.data.drop(["Extracurricular Activities","Performance Index"],axis=1).values
        self.actual_result = self.data["Performance Index"].values
        self.learn_rt = lr
        self.scalar = StandardScaler()
        self.mult_attrb = self.scalar.fit_transform(self.features)
        self.mult_attrb = np.c_[np.ones(self.mult_attrb.shape[0]),self.mult_attrb]
        self.n_samples,self.n_features = self.mult_attrb.shape

    def fit(self,epochs=300) -> np.ndarray:

        wgt = np.zeros(self.n_features)
        for _ in range(epochs):
            wgt = self.grad_desc(wgt)
        self.wgt = wgt
    
    def grad_desc(self,wgt : np.ndarray) -> np.ndarray:

        pred_result = self.mult_attrb.dot(wgt)
        error = pred_result - self.actual_result
        gradient = (2/self.n_samples) * self.mult_attrb.T.dot(error)
        new_wgt = wgt - self.learn_rt * gradient
        return new_wgt
    
    def predict(self,new_mult_attrb : pd.DataFrame) -> np.ndarray:
        new_mult_attrb = self.scalar.transform(new_mult_attrb)
        new_mult_attrb = np.c_[np.ones(new_mult_attrb.shape[0]),new_mult_attrb]
        return new_mult_attrb.dot(self.wgt)


data = pd.read_csv("Use/the/csv/file.path.csv")
mlr = MLR_Stud_Performance(data,0.01)

mlr.fit(epochs=300)
print(mlr.predict([[7,99,9,1]]))
[6,96,9,0],
[9,74,7,6],
[1,85,5,6],
[3,61,6,3]]))
