#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the training and test sets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#checking the size of the data sets
print('size of training set',train.shape)
print('size of testing set',test.shape)

#top 5 and botton 5 observations
print(train.head())
print(train.tail())

#choosing the style of the graphs
plt.style.use('ggplot')
plt.rcParams['figure.figsize']=(10,6)

#plt.style.available  (to check what all graphs are available)

#Describes the nature of SlaePrice
print(train.SalePrice.describe())

print("/n")

#checking how much it differs(skewed) from the original
print(train.SalePrice.skew())
plt.hist(train.SalePrice,color='blue')
plt.show()

print("\n")


#convert them to logarithmic form to reduce the skewness of the data
target = np.log(train.SalePrice)
print('new skew',target.skew())
plt.hist(target,color='blue')
plt.show()

#filtering out numeric features from categorical features
numeric_features = train.select_dtypes(include=[np.number])
corr = numeric_features.corr()

#checking which features are more corelated to SalePrice(Top 5 and bottom 5) 
print(corr['SalePrice'].sort_values(ascending = False)[:5])
print(corr['SalePrice'].sort_values(ascending = False)[-5:])

#This map will show the most correlated features
#checking the heatmap for most effective feature on saleprice
import seaborn as sb
sb.heatmap(corr,square=True)
sb.show()

#it shows garage area has highest correlation so we can work with that.

plt.scatter(x=train.GarageArea, y=target)
plt.ylabel('target')
plt.xlabel('GarageArea')
plt.show()

#removing the additiona data points which can disturb the regression line by dragging it towards them
#basically you remove the data points in this particular set which are > 1200 on X-axis
train = train[train['GarageArea'] < 1200]
plt.scatter(x=train['GarageArea'], y=target)
plt.xlim(-200,1600)
plt.ylabel('target')
plt.xlabel('GarageArea')
plt.show()

#counting how many nulls values per featres
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

#encoding the data 
def encode(x):return 1 if x=='Partial' else 0
train['enc_condition']= train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)

#creating a pivot graph showing median sale price and conditon
condition_pivot = train.pivot_table(index='enc_condition',values='SalePrice',aggfunc=np.median)
condition_pivot.plot(kind = 'bar',color='blue')
plt.xlabel('enc_conditionn')
plt.ylabel('Median Sale price')
plt.xticks(rotation=0)
plt.show()

#replacing nullvalues by the mean of the data .interpolate() will do this tast
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
print(sum(data.isnull().sum() != 0))

X = data.drop(['SalePrice','Id'],axis=1)
Y = np.log(train.SalePrice)

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=42)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#you can check with the below algorithms also
#Random forest
#from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor()

#decision tree
#from sklearn.tree import DecisionTreeRegressor
#regressor = DecisionTreeRegressor()

#SVR
#from sklearn.svm import SVR
#regressor = SVR()



model = regressor.fit(X_train,Y_train)

model.score(X_test,Y_test)

predictions = model.predict(X_test)

#checking mean squared error
from sklearn.metrics import mean_squared_error as mse
mse(Y_test,predictions)



