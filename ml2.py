import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as px 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 

#load data
df=pd.read_csv('data/uber.csv')
df.head()

df.columns

#col Unnamed: 0 & key is not important so i will drop it
df.drop(columns='Unnamed: 0',inplace=True)
df.drop(columns='key',inplace=True)


px.histogram(df,x='fare_amount',width=700,height=500)
df[df['fare_amount'].values<=0]

df.drop(df[df['fare_amount'].values<=0].index,inplace=True)


plt.boxplot(df['passenger_count'])


df.drop(df[df['passenger_count']>6].index,inplace=True)

df[df['passenger_count']>6]

px.box(df,x='pickup_latitude',width=700,height=500)

def filter_latitude(val):
    if val<-90 or val >90:
        return np.nan 
    else:
        return val 

def filter_longitude(val):
    if val <-180 or val >180:
        return np.nan
    else:
        return val 

df['pickup_longitude']=df['pickup_longitude'].apply(filter_longitude)
df['pickup_latitude']=df['pickup_latitude'].apply(filter_latitude)

df['dropoff_longitude']=df['dropoff_longitude'].apply(filter_longitude)
df['dropoff_latitude']=df['dropoff_latitude'].apply(filter_latitude)
df.dropna(inplace=True)


# calculate distance between pickup and dropoff using geopy liabrary
from geopy.distance import great_circle

def distance_km(x):
    pickup=(x['pickup_latitude'],x['pickup_longitude'])
    dropoff=(x['dropoff_latitude'],x['dropoff_longitude'])
    return great_circle(pickup,dropoff).km



df['distance_km'] = df.apply(lambda x: distance_km(x), axis=1) 



df.drop(['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude'] , inplace=True , axis= 1)

df[df['distance_km']==0]

df.drop(df[df['distance_km']==0].index , inplace= True )

## handileng the outliers with pandas
q1 = df['distance_km'].quantile(0.25)
q3 = df['distance_km'].quantile(0.75)
iqr = q3 - q1
lower =  q1 - 1.5 * iqr
upper = q3 + 1.5 *iqr


df_clean0 = df[(df['distance_km'] >= lower) & (df['distance_km']<=upper) ]


df_clean0.drop(df_clean0[df_clean0['distance_km'] < 1].index, inplace=True)

## handileng the outliers with pandas
q1 = df['fare_amount'].quantile(0.25)
q3 = df['fare_amount'].quantile(0.75)
iqr = q3 - q1
lower =  q1 - 1.5 * iqr
upper = q3 + 1.5 *iqr

df_clean = df_clean0[(df_clean0['fare_amount'] >= lower) & (df_clean0['fare_amount']<=upper) ]

df_clean.duplicated().sum()


# convert the col pickup_datetime type to date and split it to Year and Month 
df_clean['pickup_datetime'] = pd.to_datetime(df_clean['pickup_datetime'] ) 

# get the day, weekday, month, year, hour from pickup_datetime
df_clean['day'] = df_clean['pickup_datetime'].dt.day_name()
df_clean['weekday'] = df_clean['pickup_datetime'].dt.weekday
df_clean['month'] = df_clean['pickup_datetime'].dt.month_name()
df_clean['year'] = df_clean['pickup_datetime'].dt.year
df_clean['hour'] = df_clean['pickup_datetime'].dt.hour

# drop pickup_datetime
df_clean.drop('pickup_datetime', axis=1, inplace=True)

df_clean.drop(df_clean[df_clean['fare_amount'] > 200].index, inplace=True)
df_clean.drop(df_clean[df_clean['fare_amount'] < 1].index, inplace=True)
sns.set_style(style='darkgrid')
sns.distplot(df_clean['fare_amount'], hist=True, kde=True) 


# data spliting
x = df_clean.drop('fare_amount', axis=1)
y = df_clean['fare_amount']


# Splitting the dataset into the Train set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

cat_cols = x_train.select_dtypes(include='O').columns.tolist()
num_cols = x_train.select_dtypes(exclude='O').columns.tolist()






from sklearn.preprocessing import  RobustScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps = [
    ('scaler' , RobustScaler())
])
f
#preprocessing for categorical data 
categorical_transformer = Pipeline(steps=[
    ('enc',OneHotEncoder())
])

# Bundle preprocessing for numerical and categorical data
preprocessing = ColumnTransformer(
    transformers=[
        ('num' , numerical_transformer , num_cols),
        ('cat',categorical_transformer ,cat_cols)
    ]

)


x_train = preprocessing.fit_transform(x_train)
x_test = preprocessing.transform(x_test)


# linear regression 
from sklearn.linear_model import LinearRegression 
lr = LinearRegression()
lr.fit(x_train, y_train)

# Model Evaluation
from sklearn.metrics import mean_squared_error, r2_score

# Predictions
y_pred = lr.predict(x_test)

# Model Evaluation
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2:', r2_score(y_test, y_pred))


# Ridge 
from sklearn.linear_model import Ridge ,Lasso
r = Ridge()
r = r.fit(x_train , y_train)

y_pred_ridge = r.predict(x_test) 
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
print('R2:', r2_score(y_test, y_pred_ridge))


# Initialize the Lasso Model
lasso_reg = Lasso(alpha=1.0)
lasso_reg.fit(x_train, y_train)
lasso_pred = lasso_reg.predict(x_test)
print('RMSE:', np.sqrt(mean_squared_error(y_test, lasso_pred)))
print('R2:', r2_score(y_test, lasso_pred))






