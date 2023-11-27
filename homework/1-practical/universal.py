import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

import scipy.stats as st
from scipy.stats import shapiro
from scipy.stats import norm

import sklearn 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

def get_ASY(y_pred, y_targ, p=[-1, 1, 1, -1]):
    TP = np.sum((y_pred == 1) & (y_targ == 1))
    FP = np.sum((y_pred == 1) & (y_targ == 0))
    FN = np.sum((y_pred == 0) & (y_targ == 1))
    TN = np.sum((y_pred == 0) & (y_targ == 0))
    return p[0]*TP + p[1]*FP + p[2]*FN + p[3]*TN
    
class Preparator:
    def __init__(self):
        pass
    
    def set_dataset(self, num):
        self.X = pd.read_csv(f'./data/task1_{num}_learn_X.csv', header=None, sep=' ')
        self.y = pd.read_csv(f'./data/task1_{num}_learn_y.csv', header=None, sep=' ')
        self.X_test = pd.read_csv(f'./data/task1_{num}_learn_X.csv', header=None, sep=' ')
        print(f"0 = {(1.0 - sum(self.y.values)/len(self.y.values))[0]} | 1 = {(sum(self.y.values)/len(self.y.values))[0]}")
        
    def draw_heatmap(self, test=False):
        tmp = self.X.copy()
        tmp['answer'] = self.y.values
        sns.heatmap(tmp.corr(), cmap="viridis")
        plt.show()
        if test:
            sns.heatmap(self.X_test.corr(), cmap="viridis")
            plt.show()
    
    def calculate(self):
        pass
    
    def calculate_metrics(self, model, x_test, y_test, thr_acc, thr_asy1, thr_asy2):
        y_prob      = model.predict_proba(x_test)
        y_pred_acc  = (y_prob[:, 1] > thr_acc).astype(int)
        y_pred_asy1 = (y_prob[:, 1] > thr_asy1).astype(int)
        y_pred_asy2 = (y_prob[:, 1] > thr_asy2).astype(int)
        
        auc  = roc_auc_score(y_test, y_prob[:, 1])
        
        acc  = accuracy_score(y_test, y_pred_acc)
        asy1 = get_ASY(y_pred_asy1, y_test, p=[-9, 9, 1, 0])
        asy2 = get_ASY(y_pred_asy2, y_test, p=[-1, 3, 2, -1])
        print(f"AUC = {round(auc, 4)} | ACC = {round(acc, 4)} | ASY1 = {asy1} | ASY2 = {asy2}")
    
    def remove_outliers(self, alpha=0.05, shapiro_p0=0.999, shapiro_p1=0.999, th_left=4, th_right=4):
        mask0 = ((self.y==0).values)[:, 0]
        mask1 = ((self.y==1).values)[:, 0]
        
        for col in self.X.columns:
            x0 = self.X[col][mask0].values
            x1 = self.X[col][mask1].values
            if (shapiro(x0)[0] > shapiro_p0 and shapiro(x1)[0] > shapiro_p1):
                continue
            
            thress = (self.X[col].quantile(0.5*alpha), self.X[col].quantile(1-0.5*alpha))
    
            # Remove by 5 percent percentile
            col_vals = self.X[col].values
            mask = self.X[col].apply(lambda x: x<thress[1] and x>thress[0]).values
    
            col_filt_vals = col_vals[mask]
    
            # Get new 
            mu, std = norm.fit(col_filt_vals)
            new_thress = (mu - th_left*std, mu + th_right*std)
        
            self.X[col] = self.X[col].apply(lambda x: new_thress[0] if x<new_thress[0] else \
                                                      new_thress[1] if x>new_thress[1] else \
                                                      x).values
    def apply_transform(self):
        # 14 column
        self.X = self.X[[2, 31, 75, 84]]
        
    def draw_KLdiv(self):
        kulback = []
        mask0 = ((self.y==0).values)[:, 0]
        mask1 = ((self.y==1).values)[:, 0]
        for col in self.X.columns:
            x0 = self.X[col][mask0].values
            x1 = self.X[col][mask1].values
            shapiro_p0 = float(shapiro(x0)[0])
            shapiro_p1 = float(shapiro(x1)[0])
            mu0, std0 = norm.fit(x0)
            mu0, std0 = float(mu0), float(std0)
            mu1, std1 = norm.fit(x1)
            mu1, std1 = float(mu1), float(std1)
            kl = float(np.log(std1/std0) + (std0**2 + (mu0 - mu1)**2)/(2 * std1**2) - 0.5)
            
            print(f"{col:02d} | KL_div: {kl:3.3f} | Shapiro_ps: ({shapiro_p0:3.3f}, {shapiro_p1:3.3f}) | N({mu0:7.3f}, {std0:7.3f}) | N({mu1:6.3f}, {std1:6.3f})")
            kulback.append(kl)
        return kulback