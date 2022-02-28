
import pandas as pd
import numpy as np
import catboost as cb
import shap
import matplotlib.pyplot as plt; plt.style.use('seaborn')
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#################################特征选择部分

data = pd.read_csv('1.csv')
cols = data.columns[1:730]

model = cb.CatBoostClassifier(iterations=1000, depth=5, learning_rate=0.05, loss_function='CrossEntropy',
                              logging_level='Verbose')

label = ['Caco-2','CYP3A4','hERG','HOB','MN']
def get_shap_importance(label_name,model):

    model.fit(data[cols], data[label_name].values)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data[cols], data['Caco-2'].values)   #shape值
    importances = model.feature_importances_
    feat_names = cols
    importances_indices = np.argsort(importances)[::-1]

    return shap_values,importances,importances_indices


A_1 = pd.read_csv('top20_norm.csv')
shap_values,importances ,importances_indices= get_shap_importance('Caco-2',model)#'Caco-2'
A = pd.DataFrame(cols[importances_indices])#'Caco-2'
shap_values,importances ,importances_indices= get_shap_importance('CYP3A4',model)#'CYP3A4'
B = pd.DataFrame(cols[importances_indices])#'CYP3A4'
shap_values,importances ,importances_indices= get_shap_importance('hERG',model)#'CYP3A4'
C = pd.DataFrame(cols[importances_indices])#'hERG'
shap_values,importances ,importances_indices= get_shap_importance('HOB',model)#'CYP3A4'
D = pd.DataFrame(cols[importances_indices])#'HOB'
shap_values,importances ,importances_indices= get_shap_importance('MN',model)#'CYP3A4'
E = pd.DataFrame(cols[importances_indices])#'MN'

def gx(m):
    """

    :param m:
    :return: 在第一问20个特征中，对于第三问分类模型特征重要性排序后的前m个特征的共有特征
    """
    n = 0
    l1 = []
    for i in range(m):
        if A[0][i] in A_1.columns:
            n += 1
            l1.append(A[0][i])
    l2 = []
    for i in range(m):
        if B[0][i] in A_1.columns:
            n += 1
            l2.append(B[0][i])
    l3 = []
    for i in range(m):
        if C[0][i] in A_1.columns:
            n += 1
            l3.append(C[0][i])

    l4 = []
    for i in range(m):
        if D[0][i] in A_1.columns:
            n += 1
            l4.append(D[0][i])

    l5 = []
    for i in range(m):
        if E[0][i] in A_1.columns:
            n += 1
            l5.append(E[0][i])
    return l1, l2, l3, l4, l5

a,b,c,d,e= gx(50)
set(a)&set(b)&set(c),set(a)&set(b)&set(d),set(a)&set(b)&set(e),set(a)&set(c)&set(d),set(a)&set(c)&set(e),set(a)&set(d)&set(e),set(b)&set(c)&set(d),set(b)&set(c)&set(e),set(b)&set(d)&set(e),set(c)&set(d)&set(e)



##确定得到如下描述符
search = ['LipoaffinityIndex','MDEC-23','minHsOH','SsOH']
for i in label:
    shap_values, importances, importances_indices = get_shap_importance('Caco-2', model)
    pd.DataFrame(shap_values,columns = cols).loc[:,search].to_csv(str(i)+'search_4_shap.csv')
    ft = pd.DataFrame(cols[importances_indices])  # .to_csv('3_5.csv')



##########可视化

shap.summary_plot(shap_values, data[cols],layered_violin_max_num_bins=30)   #shap绘图

for i in label:
    shap_values, importances, importances_indices = get_shap_importance(i, model)
    m = 40
    plt.figure(figsize=(16, 9))  # figsize=(16, 9)
    shap.summary_plot(shap_values, data[cols], layered_violin_max_num_bins=30)  # shap绘图
    # plt.title('catboost算法计算出的与留存相关特征重要性')
    plt.barh(cols[importances_indices[:m]][::-1], importances[importances_indices[:m]][::-1], height=0.5)
    # plt.show()
    plt.savefig(str(i)+'features_stored.png')





