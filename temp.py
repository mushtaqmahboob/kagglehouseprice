import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print('size of training set',train.shape)
print('size of testing set',test.shape)

print(train.head())
print(train.tail())

plt.style.use('ggplot')
plt.rcParams['figure.figsize']=(10,6)

#plt.style.available

print(train.SalePrice.describe())

print("\n")

print(train.SalePrice.skew())
plt.hist(train.SalePrice,color='blue')
plt.show()

print("\n")

target = np.log(train.SalePrice)
print('new skew',target.skew())
plt.hist(target,color='blue')
plt.show()


numeric_features = train.select_dtypes(include=[np.number])
corr = numeric_features.corr()

print(corr['SalePrice'].sort_values(ascending = False)[:5])
print(corr['SalePrice'].sort_values(ascending = False)[-5:])

plt.scatter(x=train.GarageArea, y=target)
plt.ylabel('target')
plt.xlabel('GarageArea')
plt.show()

train = train[train['GarageArea'] < 1200]
plt.scatter(x=train['GarageArea'],y=target)
plt.xlim(-200,1600)
plt.ylabel('target')
plt.xlabel('GarageArea')
plt.show()


nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending = False) [:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Features'
print(nulls)

#viewing the categoricaldata (non numeric)

categoricals = train.select_dtypes(exclude = [np.number])
print(categoricals.describe()) 

print(train.Street.value_counts())

train['enc_street'] = pd.get_dummies(train.Street,drop_first = True)
test['enc_Street'] = pd.get_dummies(test.Street,drop_first = True)


print(train.enc_street.value_counts())

condition_pivot = train.pivot_table(index='SaleCondition',values='SalePrice',aggfunc=np.median)
condition_pivot.plot(kind = 'bar',color='blue')
plt.xlabel('SaleCondition')
plt.ylabel('Median Sale price')
plt.xticks(rotation=0)
plt.show()

def encode(x):return 1 if x=='Partial' else 0
train['enc_condition']= train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)

condition_pivot = train.pivot_table(index='enc_condition',values='SalePrice',aggfunc=np.median)
condition_pivot.plot(kind = 'bar',color='blue')
plt.xlabel('enc_conditionn')
plt.ylabel('Median Sale price')
plt.xticks(rotation=0)
plt.show()

data = train.select_dtypes(include=[np.number]).interpolate().dropna()
print(sum(data.isnull().sum() != 0))

X = data.drop(['SalePrice','Id'],axis=1)
Y = np.log(train.SalePrice)

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=42)

#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()

#Random forest
#from sklearn.ensemble import RandomForestRegressor
#reg = RandomForestRegressor()

#decision tree
from sklearn.tree import DecisionTreeRegressor
regg = DecisionTreeRegressor()

model = regg.fit(X_train,Y_train)

model.score(X_test,Y_test)

predictions = model.predict(X_test)

from sklearn.metrics import mean_squared_error as mse
mse(Y_test,predictions)


