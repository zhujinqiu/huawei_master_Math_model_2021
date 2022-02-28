from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import pandas as pd
import numpy as np
import catboost as cb
from sklearn import linear_model

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
train  = pd.read_csv('D:\huawei\\top20_norm.csv')
y_train = pd.read_csv('D:\huawei\ER_activity.csv')
train_no_norm = pd.read_csv(r'D:\huawei\train_原始.csv')
train_no_norm_20f = train_no_norm[train.columns]

X_train, X_test, Y_train, Y_test =train_test_split(train, y_train, test_size=0.2, shuffle=True)

#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=1).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

###################基学习器
weak = ['lin_Reg','svr_lin_reg','svr_rbf_reg','dt_reg','knn_reg','GP_reg']
def get_weak_rmse(weak,X_train,Y_train,X_test,Y_test):
    weak_rmse = []
    r2 = []
    weak_rmse_k_fold = []
    for name in weak:
        if name== 'lin_Reg':
            lr = linear_model.LinearRegression()
            y_pred = lr.fit(X_train, Y_train).predict(X_test)
            weak_rmse.append(np.sqrt(mean_squared_error(y_pred, Y_test)))
            r2.append(r2_score(y_pred, Y_test))
            # weak_rmse_k_fold.append(np.mean(rmsle_cv(lr)))
        if name == 'svr_lin_reg':
            svr_lin = SVR(kernel='linear')
            y_pred = svr_lin.fit(X_train, Y_train).predict(X_test)
            weak_rmse.append(np.sqrt(mean_squared_error(y_pred, Y_test)))
            r2.append(r2_score(y_pred, Y_test))
            # weak_rmse_k_fold.append(np.mean(rmsle_cv(svr_lin)))
        if name == 'svr_rbf_reg':
            svr_rbf = SVR(kernel='rbf')
            y_pred = svr_rbf.fit(X_train, Y_train).predict(X_test)
            weak_rmse.append(np.sqrt(mean_squared_error(y_pred, Y_test)))
            r2.append(r2_score(y_pred, Y_test))
            # weak_rmse_k_fold.append(np.mean(rmsle_cv(svr_rbf)))
        if name == 'dt_reg':
            DTR = DecisionTreeRegressor(max_depth = 5)
            y_pred = DTR.fit(X_train, Y_train).predict(X_test)
            weak_rmse.append(np.sqrt(mean_squared_error(y_pred, Y_test)))
            r2.append(r2_score(y_pred, Y_test))
            # weak_rmse_k_fold.append(np.mean(rmsle_cv(DTR)))
        if name == 'knn_reg':
            knn = KNeighborsRegressor(n_neighbors=6)
            y_pred =knn.fit(X_train, Y_train).predict(X_test)
            weak_rmse.append(np.sqrt(mean_squared_error(y_pred, Y_test)))
            r2.append(r2_score(y_pred, Y_test))
            # weak_rmse_k_fold.append(np.mean(rmsle_cv(knn)))
        if name == 'GP_reg':
            kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
            GPR = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
            y_pred = GPR.fit(X_train, Y_train).predict(X_test)
            weak_rmse.append(np.sqrt(mean_squared_error(y_pred, Y_test)))
            r2.append(r2_score(y_pred, Y_test))
            # weak_rmse_k_fold.append(np.mean(rmsle_cv(GPR)))
    return pd.DataFrame([weak_rmse,r2],columns = weak,index = ['rmse','R2'])#,pd.DataFrame([weak_rmse_k_fold],columns = weak,index = ['rmse'])

weak_rmse = get_weak_rmse(weak,X_train,Y_train,X_test,Y_test)



###################强学习器
strong = ['RF','GBRT',"XGboost","Catboost","lightgbm"]
def get_strong_rmse_R2(strong,X_train,Y_train,X_test,Y_test,n_estimators):
    strong_rmse = []
    strong_rmse_k_fold = []
    r2 = []
    for name in strong:
        if name == 'RF':
            rf = RandomForestRegressor(n_estimators=n_estimators,random_state=0,min_samples_leaf=10)
            y_pred = rf.fit(X_train, Y_train).predict(X_test)
            strong_rmse.append(np.sqrt(mean_squared_error(y_pred, Y_test)))
            r2.append(r2_score(y_pred, Y_test))
            # strong_rmse_k_fold.append(np.mean(rmsle_cv(rf)))
        if name == 'GBRT':
            GBoost = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=0.1,
                                               max_depth=20, max_features='sqrt',
                                               min_samples_leaf=10, min_samples_split=10,
                                               loss='huber', random_state=1)
            y_pred = GBoost.fit(X_train, Y_train).predict(X_test)
            strong_rmse.append(np.sqrt(mean_squared_error(y_pred, Y_test)))
            r2.append(r2_score(y_pred, Y_test))
            # strong_rmse_k_fold.append(np.mean(rmsle_cv(GBoost)))
        if name == 'XGboost':
            model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                                         learning_rate=0.05, max_depth=8,
                                         min_child_weight=1.7817, n_estimators=n_estimators,
                                         reg_alpha=0.5, reg_lambda=0.8571,
                                         subsample=0.5213,
                                         random_state=1, nthread=-1)
            y_pred = model_xgb.fit(X_train, Y_train).predict(X_test)
            strong_rmse.append(np.sqrt(mean_squared_error(y_pred, Y_test)))
            r2.append(r2_score(y_pred, Y_test))
            # weak_rmse_k_fold.append(np.mean(rmsle_cv(svr_rbf)))
        if name == 'Catboost':
            cat = cb.CatBoostRegressor(
                iterations=1000,random_state=1)
            y_pred = cat.fit(X_train, Y_train).predict(X_test)
            strong_rmse.append(np.sqrt(mean_squared_error(y_pred, Y_test)))
            r2.append(r2_score(y_pred, Y_test))
            # weak_rmse_k_fold.append(np.mean(rmsle_cv(DTR)))

        if name == 'lightgbm':
            lgb = LGBMRegressor(random_state=1,n_estimators=n_estimators)
            y_pred = lgb.fit(X_train, Y_train).predict(X_test)
            strong_rmse.append(np.sqrt(mean_squared_error(y_pred, Y_test)))
            r2.append(r2_score(y_pred, Y_test))
            # weak_rmse_k_fold.append(np.mean(rmsle_cv(knn)))
    return pd.DataFrame([strong_rmse,r2],columns = strong,index = ['rmse','R2'])
get_strong_rmse_R2(strong,X_train,Y_train,X_test,Y_test,150)

# score = rmsle_cv(reg)
# print("\n reg score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

X_train, X_test, Y_train, Y_test =train_test_split(train, y_train, test_size=0.2, shuffle=True)

###################强学习器—bagging
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                                         learning_rate=0.05, max_depth=8,
                                         min_child_weight=1.7817, n_estimators=160,
                                         reg_alpha=0.5, reg_lambda=0.8571,
                                         subsample=0.5213,
                                         random_state=1, nthread=-1)
cat = cb.CatBoostRegressor(
    iterations=1000, random_state=1)
lgb = LGBMRegressor(random_state=0, n_estimators=160)
GBoost = GradientBoostingRegressor(n_estimators=160, learning_rate=0.1,
                                               max_depth=10, max_features='sqrt',
                                               min_samples_leaf=10, min_samples_split=10,
                                               loss='huber', random_state=1)
rf = RandomForestRegressor(n_estimators=100,random_state=1,min_samples_leaf=10)


test_no_norm = pd.read_csv(r'D:\huawei\test_q2.csv')
test_norm_self = pd.read_csv(r'D:\huawei\test_q2_norm_自己归一化.csv')
test_norm_all = pd.read_csv(r'D:\huawei\test_q2_norm_与原始数据归一化.csv')
#投票集成，取均值
from sklearn.ensemble import VotingRegressor
# reg0 = rf
reg1 = model_xgb  #0.68
reg2 = GBoost   #0.68
reg3 = cat   # 0.69
reg4 = lgb
# reg5 = DecisionTreeRegressor()
# reg6 = DecisionTreeRegressor()
# reg3 = DecisionTreeRegressor()
ereg = VotingRegressor(estimators=[('reg1', reg1),('reg2', reg2),('reg3', reg3),('reg4', reg4)]
                       ,weights= [0.25308791, 0.24482512, 0.24854233, 0.25354463])#,('reg3', reg3)weights = [0.24974171, 0.24674952, 0.24995223, 0.25355654]
ereg  = ereg.fit(train_no_norm_20f,y_train)
y_pre_norm_sefl = ereg.predict(test_no_norm[train.columns])
# y_pre_norm_self= ereg.predict(test_norm_self[train.columns])

# pre_no_norm  = ereg.predict(test_no_norm[train.columns])

# ereg.fit(train,y_train).predict(test_no_norm[train.columns])
print(np.sqrt(mean_squared_error(y_pre,y_train)))
print(r2_score(y_pre,y_train))


import joblib
# joblib.dump(ereg, 'norm_q2_model.pkl')
joblib.dump(ereg,'no_norm_q2_model.pkl')

#加载模型
model_name= joblib.load('no_norm_q2_model.pkl')

ss =model_name.predict(train_no_norm_20f)


print(np.sqrt(mean_squared_error(ss,y_train)))
print(r2_score(ss,y_train))


###################强学习器-stacking
from sklearn.linear_model import RidgeCV, LassoCV
# from sklearn.neighbors import KNeighborsRegressor
estimators = [('xgboost', reg1),
                ('cat', reg3),
              ('gbdt', reg2),
                ('lgb', reg4)
              ]
# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
final_estimator = LassoCV()
reg = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator)
print(np.sqrt(mean_squared_error(reg.fit(train, y_train).predict(train), y_train)))
print(r2_score(reg.fit(train, y_train).predict(train), y_train))
# score = rmsle_cv(reg)
# print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


train_no_norm_20f.to_csv('前20个特征无归一化训练集.csv')