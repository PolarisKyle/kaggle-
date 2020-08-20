import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#加载相关库
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor

df_train = pd.read_csv('E:\WorkPy\house-prices\data\\train.csv')
df_test = pd.read_csv('E:\WorkPy\house-prices\data\\test.csv')

#数据清洗
df_train.shape,df_test.shape
y_train=df_train.pop('SalePrice')  #删除并返回数据集中SalePrice标签列
all_df=pd.concat((df_train,df_test),axis=0) #要处理的整体数据集


total=all_df.isnull().sum().sort_values(ascending=False)  #每列缺失数量
percent=(all_df.isnull().sum()/len(all_df)).sort_values(ascending=False) #每列缺失率
miss_data=pd.concat([total,percent],axis=1,keys=['total','percent'])
miss_data #显示每个列及其对应的缺失率

all_df=all_df.drop(miss_data[miss_data['percent']>0.4].index,axis=1) #去除了percent>0.4的列

garage_obj=['GarageType','GarageFinish','GarageQual','GarageCond'] #列出车库这一类
for garage in garage_obj:
   all_df[garage].fillna('missing',inplace=True)

#把1900标签填入空缺处表示年代久远
all_df['GarageYrBlt'].fillna(1900.,inplace=True)


all_df['MasVnrType'].fillna('missing',inplace=True)  #用missing标签表示没装修过
all_df['MasVnrArea'].fillna(0,inplace=True)   #用0表示没装修过的装修面积

#再次查看数据缺失率，最高为0.16，是LotFrontage列
(all_df.isnull().sum()/len(all_df)).sort_values(ascending=False)

#从图中看出LotFrontage分布较均匀，可以用均值补齐缺失值
plt.figure(figsize=(16,6))
plt.plot(all_df['Id'],all_df['LotFrontage'])
plt.show()

#均值补齐LotFrontage列
all_df['LotFrontage'].fillna(all_df['LotFrontage'].mean(),inplace=True)

#还有部分少量的缺失值，不是很重要，可以用one-hotd转变离散值，然后均值补齐
all_dummies_df=pd.get_dummies(all_df)
mean_col=all_dummies_df.mean()
all_dummies_df.fillna(mean_col,inplace=True)

#数据集中数值类型为int和float
all_dummies_df['Id']=all_dummies_df['Id'].astype(str)  #先排除ID列，不对Id列进行处理
a=all_dummies_df.columns[all_dummies_df.dtypes=='int64'] #数值为int型
b=all_dummies_df.columns[all_dummies_df.dtypes=='float64'] #数值为float型

#进行标准化处理，符合0-1分布
a_mean=all_dummies_df.loc[:,a].mean()
a_std=all_dummies_df.loc[:,a].std()
all_dummies_df.loc[:,a]=(all_dummies_df.loc[:,a]-a_mean)/a_std #使数值型为int的所有列标准化
b_mean=all_dummies_df.loc[:,b].mean()
b_std=all_dummies_df.loc[:,b].std()
all_dummies_df.loc[:,b]=(all_dummies_df.loc[:,b]-b_mean)/b_std #使数值型为float的所有列标准化


 #处理后的训练集(不含Saleprice)
df_train1=all_dummies_df.iloc[:1460,:]

df_train_train=df_train1.iloc[0:int(0.8*len(df_train1)),:]  #train中的训练集(不含Saleprice)
df_train_test=df_train1.iloc[int(0.8*len(df_train1)):,:]    #train中的测试集(不含Saleprice)

df_train_train_y=y_train.iloc[0:int(0.8*len(y_train))]     #train中训练集的target
df_train_test_y=y_train.iloc[int(0.8*len(df_train1)):]     #train中测试集的target

#处理后的测试集
df_test1=all_dummies_df.iloc[1460:,:]

max_features=[.1,.2,.3,.4,.5,.6,.7,.8,.9]
test_score=[]
for max_feature in max_features:
    clf=RandomForestRegressor(max_features=max_feature,n_estimators=100)
    score=np.sqrt(cross_val_score(clf,df_train_train,df_train_train_y,cv=5))
    test_score.append(1-np.mean(score))

plt.plot(max_features,test_score) #得出误差得分图
plt.show()

rf=RandomForestRegressor(max_features=0.5,n_estimators=100)
rf.fit(df_train_train,df_train_train_y)
#用均方误差来判断模型好坏，结果越小越好
aa = (((df_train_test_y-rf.predict(df_train_test))**2).sum())/len(df_train_test_y)
print(aa)
bb = rf.predict(df_test)
print(bb)
print(bb.shape)
# sampleSubmission = pd.read_csv('E:\WorkPy\house-prices\data\sample_submission.csv')
# sampleSubmission['SalePrice'] = rf.predict(df_train_test)
# sampleSubmission.to_csv('E:\\WorkPy\\house-prices\\data\\result.csv',index = False)