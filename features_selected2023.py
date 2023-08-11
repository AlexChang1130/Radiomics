#!/usr/bin/env python
# coding: utf-8

# ### 导入包

# In[31]:


import pandas as pd
import numpy as np
from collections import Counter
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy.stats import mannwhitneyu, fisher_exact
from sklearn.metrics import classification_report, plot_roc_curve
from imblearn.over_sampling import SMOTE


# ###

# In[32]:


dataPath = r'C:\Users\ZCS\Desktop\30%\30ctyxzx.xlsx'
data = pd.read_excel(dataPath)
data.describe()


# In[33]:


data.info()




# In[34]:


y = data['Label']
Counter(y)


# In[35]:


data_a = data[y == 0]
data_b = data[y == 1]
data_a.shape, data_b.shape


# In[36]:


X = data.iloc[:,1:]
X.head()
X.shape


# ### 划分训练集、测试集

# In[37]:


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21, stratify = y)
X_train = X[47:]
y_train = y[47:]
X_test = X[:47]
y_test = y[:47]
X_train.to_excel(r'C:\Users\ZCS\Desktop\01\0423train.xlsx',sheet_name='train',header=True)
X_test.to_excel(r'C:\Users\ZCS\Desktop\01\0423test.xlsx',sheet_name='test',header=True)


# In[38]:


X_train_a = X_train[y_train == 0]
X_train_b = X_train[y_train == 1]


# In[39]:


X_train_a.head()


# ### 

# In[40]:


colNamesSel_mwU = []
for colName in X_train_a.columns[:]: 
    try:
        if mannwhitneyu(X_train_a[colName],X_train_b[colName])[1] < 0.05:
            colNamesSel_mwU.append(colName)
    except:
        print(colName,'gets error !!')
print(len(colNamesSel_mwU))
print(colNamesSel_mwU)


# In[41]:


X_train_mul = X_train[colNamesSel_mwU]
X_test_mul = X_test[colNamesSel_mwU]
X_train_mul




# In[42]:


scaler = StandardScaler()
X_train_mul_scal = scaler.fit_transform(X_train_mul)
X_test_mul_scal = scaler.transform(X_test_mul)


# In[43]:


X_train_mul_scal = pd.DataFrame(X_train_mul_scal,columns = colNamesSel_mwU)
X_test_mul_scal = pd.DataFrame(X_test_mul_scal,columns = colNamesSel_mwU)
X_train_mul_scal


# ### LASSO

# In[45]:


alphas = np.logspace(-3, 1, 100)
selector_lasso = LassoCV(alphas=alphas, cv = 5, max_iter = 100000)
selector_lasso.fit(X_train_mul_scal, y_train)
print(selector_lasso.alpha_)
values = selector_lasso.coef_[selector_lasso.coef_ != 0]
colNames_sel = X_train_mul_scal.columns[selector_lasso.coef_ != 0]
print(colNames_sel)
print(data.shape)
#data[:,selector_lasso.coef_ != 0].to_excel(r'C:\Users\ZCS\Desktop\shuaixuan.xlsx',sheet_name='sx',header=True)
print(len(X_train_mul_scal.columns[selector_lasso.coef_ != 0]))


# In[46]:


features_selected = data.iloc[:,:]
features_selected =features_selected[colNames_sel]
features_selected.to_excel(r'C:\Users\ZCS\Desktop\30%\30ctyxzxsx.xlsx',sheet_name='sx',header=True)


# In[47]:


width = 0.45
plt.bar(colNames_sel, values
        , color= 'lightblue'
        , alpha = 1)
plt.xticks(np.arange(len(colNames_sel)),colNames_sel
           , rotation='60'
           , ha = 'right'
          )
plt.ylabel("Coefficient")
plt.show()


# In[ ]:





# In[48]:


MSEs_mean = selector_lasso.mse_path_.mean(axis = 1)
MSEs_std = selector_lasso.mse_path_.std(axis = 1)
plt.figure()
plt.errorbar(selector_lasso.alphas_,MSEs_mean   
             , yerr=MSEs_std                   
             , fmt="o"                     
             , ms=3                            
             , mfc="r"                          
             , mec="r"                         
             , ecolor="lightblue"              
             , elinewidth=2                     
             , capsize=4                       
             , capthick=1)                     
plt.semilogx()
plt.axvline(selector_lasso.alpha_,color = 'black',ls="--")
plt.xlim(1e-3,10)
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.show()


# In[ ]:





# In[49]:


coefs = selector_lasso.path(X_train_mul_scal, y_train, alphas=alphas, max_iter = 1000000
                           )[1].T
plt.figure()
plt.semilogx(selector_lasso.alphas_,coefs, '-')
plt.axvline(selector_lasso.alpha_,color = 'black',ls="--")
plt.xlim(1e-3,10)
plt.ylim(-0.5,0.5)
plt.xlabel('Lambda')
plt.ylabel('Coefficients')
plt.show()








