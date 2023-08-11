import xlrd
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import random
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score,classification_report
from sklearn.metrics import confusion_matrix
# multiple classifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import BernoulliRBM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV

warnings.filterwarnings('ignore')

# -----------------------------------------> definition
# Reference: https://scikit-learn.org/stable/modules/feature_selection.html

selection_names = ['linear_SVC', 'lasso', 'ExtraTree']

# classifier_names = ['LogisticRegression', 'RandomForestClassifier', 'AdaBoostClassifier', 'DecisionTreeClassifier',
#                     'GaussianNB', 'LinearDiscriminantAnalysis',
#                     'QuadraticDiscriminantAnalysis',
#                     'KNeighborsClassifier', 'SVC_rbf', 'SVC_linear']

classifier_names = ['SVC_rbf']

selection_methods = [SelectFromModel(LinearSVC(), max_features=10),
                    # VarianceThreshold(threshold=(.6 * (1 - .6))),
                    SelectFromModel(Lasso(alpha=.1), max_features=10),
                    SelectFromModel(ExtraTreesClassifier(n_estimators=100), max_features=20)]
# , max_features=10




# -----------------------------------------> load data

csv_file = pd.read_csv(r'C:\Users\ZCS\Desktop\30%\30ctyxzxsx+nb+lc.csv')
title = [col for col in csv_file]
feature_names = title[1:]
features = csv_file.values[:, 1:]
labels = csv_file.values[:, 0]

X = np.array(features)
y = np.array(labels)
y = y.astype('int')
final_auc = 0
# for i in range(0,30000)[::10]:
#     random.seed(i)
#     random.shuffle(X)
#     random.shuffle(y)
X_xfold = X[47:,:]
y_yfold = y[47:]
X_indenpent = X[:47,:]
y_indenpent = y[:47]
#y[y==3]=1
#y[y==2]=0
#y = y-2,
skf = StratifiedKFold(n_splits=3, shuffle=True)  # k folder cross-validation
ss = StandardScaler()  # data normalization
#



Cs = np.logspace(-1,3,10,base = 2)
gammas = np.logspace(-5,1,10)
param_grid = {
            'C': Cs
            , 'gamma': gammas
#             , 'kernel': ('rbf','linear')
}
GS = GridSearchCV(SVC()
                    , param_grid = param_grid
                    , cv = 5
                   )
GS.fit(X_xfold, y_yfold)
print(GS.best_params_)
C = GS.best_params_['C']
gamma = GS.best_params_['gamma']
classifiers = [
                # LogisticRegression(),
                # RandomForestClassifier(),
                # AdaBoostClassifier(),
                # Classifier(max_depth=2),
                # GaussianNB(),
                # LinearDiscriminantAnalysis(),
                # QuadraticDiscriminantAnalysis(),
                # KNeighborsClassifier(5),
                SVC(kernel="linear",C=0.5,gamma=0.00001,probability=True)]
                # SVC(gamma=2, C=1, probability=True)]

# DecisionTree




# # -------------------------------------------> visualize


def plot_feature_importance(clf, feature_name, title):
    """
    :param clf: method for feature selection
    :param feature_name: feature name
    :param title: the name of feature selection method
    :return: plot of selected features and its corresponding score.
    """
    if isinstance(clf, SelectFromModel):
        feature_importance = []
        feature_names = []
        if hasattr(clf.estimator_, 'coef_'):
            feature_importance_ = np.squeeze(abs(clf.estimator_.coef_))*0.5
        elif hasattr(clf.estimator_, 'feature_importances_'):
            feature_importance_ = np.squeeze(abs(clf.estimator_.feature_importances_))*0.5
        support = selector.get_support()
    else:
        feature_importance = []
        feature_names = []
        if hasattr(clf, 'coef_'):
            feature_importance_ = abs(clf.coef_)[0]
        elif hasattr(clf, 'feature_importances_'):
            feature_importance_ = abs(clf.feature_importances_)[0]
        else:
            feature_importance_ = np.ones(len(feature_name)) / len(feature_name)
        support = selector.get_support()

    for n, i in enumerate(feature_importance_):
        if support[n] == True:
            feature_importance.append(i)
            feature_names.append(feature_name[n])
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, np.array(feature_importance)[sorted_idx], align='center')
    plt.yticks(pos, [feature_names[idx] for idx in sorted_idx])
    plt.xlabel('Feature Weight')
    plt.title(title + '(num=%s)' % str(len(feature_names)))
    plt.show()

#

# for k, selector in enumerate(selection_methods):
#     selector.fit(X_xfold, y_yfold)
#     plot_feature_importance(selector, feature_names, selection_names[k])

# -------------------------------------------> classifier training and validation
tb = PrettyTable()
#tb.field_names = ['SELECTION/CLASSIFIER', 'AUC', 'ACC', 'SNES', 'SPEC']
tb.field_names = ['SELECTION/CLASSIFIER', 'AUC']
for j,classifier in enumerate(classifiers):
    # for k, selector in enumerate(selection_methods):
    print(classifier_names[j])
    clf = Pipeline([('normalization', ss),
                    ('classification', classifier)])
    gt = np.array([])
    y_preds = np.array([])
    score_preds = np.array([])
    
    gt_ind = 0
    y_preds_ind = 0
    score_preds_ind = 0
    for train, test in skf.split(X_xfold, y_yfold):
        X_train = X_xfold[train]
        X_test = X_xfold[test]
        clf.fit(X_train, y_yfold[train])
        y_pred = clf.predict(X_test)
        score = clf.predict_proba(X_test)[:, 1]
        y_preds = np.concatenate([y_preds, y_pred], axis=-1)
        score_preds = np.concatenate([score_preds, score], axis=-1)
        gt = np.concatenate([gt, y_yfold[test]], axis=-1)
        
        ############################################
        y_ind_pred = clf.predict(X_indenpent)
        ind_score = clf.predict_proba(X_indenpent)[:, 1]
        y_preds_ind = y_preds_ind+ y_ind_pred
        score_preds_ind = score_preds_ind+ind_score
        gt_ind = gt_ind+y_indenpent
        
    y_preds = np.array(y_preds)
    score_preds = np.array(score_preds)
    acc = accuracy_score(gt, y_preds)
    f1_score = classification_report(gt.astype('int'), y_preds)
    cm = confusion_matrix(gt, y_preds)
    TP = cm[0, 0]
    TN = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    import sklearn
    auc = sklearn.metrics.roc_auc_score(gt, y_preds, average='macro')
    tb.add_row([classifier_names[j],
                f1_score])
    print('AUC',auc)
    
    ######################################
    print('独立测试')
    y_preds_ind = np.array(y_preds_ind)//3
    score_preds_ind = np.array(score_preds_ind)/3
    gt_ind = gt_ind//3
    acc = accuracy_score(gt_ind, y_preds_ind)
    f1_score = classification_report(gt_ind.astype('int'), y_preds_ind)
    cm2 = confusion_matrix(gt_ind, y_preds_ind)
    TP = cm2[0, 0]
    TN = cm2[1, 1]
    FP = cm2[0, 1]
    FN = cm2[1, 0]
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    import sklearn
    auc2 = sklearn.metrics.roc_auc_score(gt_ind, y_preds_ind, average='macro')
    tb.add_row([classifier_names[j],
                f1_score])
    print('独立测试AUC',auc2)
    # if auc+auc2>final_auc:
    #     final_auc = auc+auc2
    #     print(i,auc,auc2)
    
    
#        tb.add_row([selection_names[k] + '/' + classifier_names[j],
#                    f1_score, round(acc, 3), round(sensitivity, 3), round(specificity, 3)])
# 
print(tb)
font2 = {
'size' : 16,'weight': 'bold',
}
    
fpr,tpr,threshold = sklearn.metrics.roc_curve(gt, score_preds,pos_label=1) ###计算真正率和假正率
fpr1,tpr1,threshold1 = sklearn.metrics.roc_curve(gt_ind, score_preds_ind,pos_label=1) ###计算真正率和假正率
plt.figure()
lw = 2
plt.figure(figsize=(7,7),dpi=800)
plt.plot(fpr, tpr, color='red',
         lw=lw, label='Cross validation ROC curve (area = %0.2f)' % auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot(fpr1, tpr1, color='blue',
         lw=lw, label='Independent testing ROC curve (area = %0.2f)' % auc2) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tick_params(labelsize=13,width=4)
ax=plt.gca();#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(0);###设置右边坐标轴的粗细a
ax.spines['top'].set_linewidth(0);####设置上部坐标轴的粗细
plt.xlabel('1-Specificity',font2)
plt.ylabel('Sensitivity',font2)
# plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


def plot_confusion_matrix(cm, savename, title='Normalized Confusion Matrix'):

    plt.figure(figsize=(3,3), dpi=300)
    np.set_printoptions(precision=0)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        # c = cm[y_val][x_val]/sum(cm[y_val])
        c = cm[y_val][x_val]
        # if c > 0.001:
        plt.text(x_val, y_val, "%d" % (c,), color='deeppink', fontsize=22, va='center', ha='center')
    font = {'family' : 'serif',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 16,
        }
    
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.gray_r)
    plt.title(title,fontsize = 22)
    cb=plt.colorbar()
    cb.ax.tick_params(labelsize=4)
#    cb.set_label('colorbar',fontdict=font) #设置colorbar的标签字体及其大小
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
#    plt.ylabel('Predict label')
#    plt.xlabel('Actual label')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'major', labelsize =14)
#    bx.tick_params(axis = 'both', which = 'major', labelsize = 24)
    plt.grid(True, which='minor', linestyle='-')
    # plt.gcf().subplots_adjust(bottom=0.2)
    
    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()


classes = [0, 1]
plot_confusion_matrix(cm, 'confusion_matrix_1data.png', title='Cross validation')
plot_confusion_matrix(cm2, 'confusion_matrix_duli_data.png', title='Independent testing')
