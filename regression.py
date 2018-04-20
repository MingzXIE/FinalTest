from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Lasso

# from sklearn.lin
from sklearn import datasets
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

data = pd.read_csv('testSet.txt',sep = '\t',header=None)

data_x = data.iloc[:,:-1]
data_y = data.iloc[:,-1]
y_len = len(data_y)

print(len(data_y))

n_fold = 5 # n_fold表示要进行n折检验，默认为5
kf = KFold(y_len, n_folds=n_fold, random_state=2018)


def Clf_Test(clf, kf, n_fold):
    # 利用输入数据进行n-fold交叉判断，以平均准确率（accuracy标准）判断哪个分类算法更好
    # kf表示的输入的切分数据的集合
    # clf表示输入的分类器
    list_result = []
    # random_state = 2018,当random_state = 大于0的数时，每次随机可以保持一致
    # random_state = -1 表示完全随机
    for train_index,test_index in kf:
        # kf是根据输入的target计算的切分的 行的索引
        train_x,train_y = data_x.iloc[train_index],data_y.iloc[train_index] # 确定训练数据
        test_x,test_y = data_x.iloc[test_index],data_y.iloc[test_index] # 确定检验数据
        # 以输入的分类器拟合训练数据
        clf.fit(train_x,train_y)
        # 以测试数据进行预测
        predict_y = clf.predict(test_x)
        # 将真实值和预测值进行准确率的检验
        list_result.append(mean_squared_error(test_y,predict_y))
    # 返回每个clf的平均的准确率
    return sum(list_result)/n_fold



# clf_all以字典的形式 存储分类器
# 分类器直接采用默认参数即可
clf_all= {'tree':LinearRegression(),
          'svm':SVR(),
          'logit':Lasso(),
          'bayes':BayesianRidge()
          }

clf_all_result = [] # 存储分类器的检验结果
clf_all_label = [] # 存储分类器的name
for k, v in clf_all.items():
    # 进行循环，检测和测试
    clf_all_label.append(k)
    accuracy_tmp = Clf_Test(v, kf, n_fold)
    clf_all_result.append(accuracy_tmp)
    print(k, accuracy_tmp)
# 根据准确率判断出最好的分器
best_one = clf_all_result.index(min(clf_all_result))
best_choice = clf_all_label[best_one]
print('The accuracy of clf is:',min(clf_all_result))
print("The best clf is:" ,best_choice)
# 用全量数据，训练模型
clf_best = clf_all[best_choice]
clf_best.fit(data_x, data_y)
# 输入数据（tab分割），并用最好的分类器预测，输出预测结果
input_data = input('Pleast type u data: sep by tab:')
test_data = np.array(list(map(float,input_data.strip().split('\t')))).reshape(1, -1)
predict_0 = clf_best.predict(test_data)
print("The predict value is",predict_0[0])
