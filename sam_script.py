import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation

train = pd.read_csv('data/train.csv', parse_dates='datetime', index_col='datetime')
test = pd.read_csv('data/test.csv', parse_dates='datetime', index_col='datetime')

train['hour'] = train.index.hour
train['month'] = train.index.month
train['year'] = train.index.year - 2011
train['day_of_week'] = train.index.weekday

test['hour'] = test.index.hour
test['month'] = test.index.month
test['year'] = test.index.year - 2011
test['day_of_week'] = test.index.weekday

for col in ['casual', 'registered', 'count']:
    train['log-' + col] = train[col].apply(lambda x: np.log1p(x))

features = ['season', 'holiday', 'workingday',
            'weather', 'humidity', 'temp', 'atemp', 'windspeed',
            'hour', 'month', 'year', 'day_of_week']

rf1 = RandomForestRegressor(n_estimators=200, n_jobs=2, min_samples_split=4)
rf1.fit(train[features], train['log-casual'])
rf2 = RandomForestRegressor(n_estimators=200, n_jobs=2, min_samples_split=4)
rf2.fit(train[features], train['log-registered'])
predictions = np.expm1(rf1.predict(test[features])) + np.expm1(rf2.predict(test[features]))
df_submit = pd.DataFrame(data=[int(x) for x in predictions], index=test.index, columns=['count'])
pd.DataFrame.to_csv(df_submit, 'sam_prediction.csv')
