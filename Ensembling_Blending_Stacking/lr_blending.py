import glob 
import pandas as pd 
import numpy as np
from sklearn import metrics

from functools import partial
from scipy.optimize import fmin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression


        

def run_training(pred_df, fold):
   
    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[["lr_pred", "lr_cnt_pred", "lr_svd_pred"]].values
    xvalid = valid_df[["lr_pred", "lr_cnt_pred", "lr_svd_pred"]].values 


    scl = StandardScaler()
    xtrain = scl.fit_transform(xtrain)
    xvalid = scl.fit_transform(xvalid)


    opt = LogisticRegression()
    opt.fit(xtrain, train_df.sentiment.values)
    preds = opt.predict_proba(xvalid)[:, 1]
    auc = metrics.roc_auc_score(valid_df.sentiment.values, preds)
    print(f"{fold}, {auc}")
    
    return opt.coef_ 


if __name__ == "__main__":
    files = glob.glob("/home/hasan/Desktop/ensembling_blending_stacking/files/*.csv")

    df = None 
    for f in files:
        if df is None:
            df = pd.read_csv(f)

        else:
            temp_file = pd.read_csv(f)
            df = df.merge(temp_file, on="id", how='left')

    targets = df.sentiment.values
    
    pred_cols = ["lr_pred", "lr_cnt_pred", "lr_svd_pred"]

    coefs = []
    for j in range(5): 
        coefs.append(run_training(df, j))

    coefs = np.array(coefs)
    print(coefs)
    coefs = np.mean(coefs, axis=0)
    print(coefs)

    wt_avg = (
        coefs[0][0] * df.lr_pred.values
        + coefs[0][1] * df.lr_cnt_pred.values
        + coefs[0][2] * df.lr_svd_pred.values
    )

    print("optimal auc after finding coefs")
    print(metrics.roc_auc_score(targets, wt_avg))









