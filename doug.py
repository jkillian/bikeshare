import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV


class BikeShare():

    def __init__(self):
        data = pd.read_csv('data/train.csv', parse_dates='datetime', index_col='datetime')
        self.holdout = pd.read_csv('data/test.csv', parse_dates='datetime', index_col='datetime')

        data['hour'] = data.index.hour
        data['month'] = data.index.month
        data['year'] = data.index.year - 2011
        data['day_of_week'] = data.index.weekday

        self.holdout['hour'] = self.holdout.index.hour
        self.holdout['month'] = self.holdout.index.month
        self.holdout['year'] = self.holdout.index.year - 2011
        self.holdout['day_of_week'] = self.holdout.index.weekday

        mask = np.random.rand(len(data)) < 0.75
        self.train = data[mask]
        self.test = data[~mask]

        for col in ['casual', 'registered', 'count']:
            self.train['log-' + col] = self.train[col].apply(lambda x: np.log1p(x))

    def error(self, predictions, actual):
        err = 0.0
        for i, pred in enumerate(predictions):
            err += (np.log1p(pred)-np.log1p(actual[i]))**2
        J = (err/len(predictions))**.5
        return J

    def elastic_net(self):
        enet = ElasticNet()
        # features = ['season', 'holiday', 'workingday', 'weather', 'humidity', 'temp', 'windspeed', 'hour', 'month', 'year', 'day_of_week']
        features = ['season', 'workingday', 'weather', 'humidity', 'windspeed', 'hour', 'month', 'year', 'day_of_week']
        enet = ElasticNetCV()
        enet.fit(self.train[features], self.train['log-count'])

        return self.predict(enet, "Elastic Net", features)

    def gradient_boosting_regressor(self):
        features = ['season', 'holiday', 'workingday', 'weather',
                    'temp', 'atemp', 'humidity', 'windspeed', 'year',
                    'month', 'day_of_week', 'hour']

        gbr = GradientBoostingRegressor(n_estimators=80, learning_rate=.05, max_depth=10, min_samples_leaf=20)
        gbr.fit(self.train[features], self.train['log-count'])
        self.output_results(gbr, "gbr", features)
        return self.predict(gbr, "Gradient Boostin Regressor", features)

    def predict(self, model, model_name, features):
        results = {}
        train_predictions = model.predict(self.train[features])
        train_predictions = np.expm1(train_predictions)

        train_predictions[train_predictions < 0] = 0
        results['train'] = self.error(train_predictions, self.train['count'])

        test_predictions = model.predict(self.test[features])
        test_predictions[test_predictions < 0] = 0
        test_predictions = np.expm1(test_predictions)

        results['test'] = self.error(test_predictions, self.test['count'])
        print "{}:\n Train error: {}\n Test error: {}".format(model_name, results['train'], results['test'])
        return results

    def output_results(self, model, model_name, features):
        predictions = model.predict(self.holdout[features])
        predictions = np.expm1(predictions)
        predictions[predictions < 0] = 0
        tdf = pd.DataFrame([self.holdout.index, predictions]).transpose()
        tdf.columns = ['datetime', 'count']
        tdf.to_csv('{}.csv'.format(model_name), index=False)

if __name__ == '__main__':
    bike_share = BikeShare()
    bike_share.elastic_net()
    bike_share.gradient_boosting_regressor()
