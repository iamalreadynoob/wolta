import numpy as np


def get_score(y_test, y_pred, metrics=None, average='weighted', algo_type='clf', verbose=True):
    if metrics is None:
        if algo_type == 'clf':
            metrics = ['acc']
        elif algo_type == 'reg':
            metrics = ['sq']

    output = ''
    scores = {}

    if algo_type == 'clf':
        for metric in metrics:
            if metric == 'acc':
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_test, y_pred)
                scores[metric] = score

                if output == '':
                    output = 'Accuracy Score: {}'.format(str(score))
                else:
                    output += '\nAccuracy Score: {}'.format(str(score))

            elif metric == 'f1':
                from sklearn.metrics import f1_score
                score = f1_score(y_test, y_pred, average=average)
                scores[metric] = score

                if output == '':
                    output = 'F1 Score (weighted): {}'.format(str(score))
                else:
                    output += '\nF1 Score (weighted): {}'.format(str(score))

            elif metric == 'hamming':
                from sklearn.metrics import hamming_loss
                score = hamming_loss(y_test, y_pred)
                scores[metric] = score

                if output == '':
                    output = 'Hamming Loss: {}'.format(str(score))
                else:
                    output += '\nHamming Loss: {}'.format(str(score))

            elif metric == 'jaccard':
                from sklearn.metrics import jaccard_score
                score = jaccard_score(y_test, y_pred, average=average)
                scores[metric] = score

                if output == '':
                    output = 'Jaccard: {}'.format(str(score))
                else:
                    output += '\nJaccard: {}'.format(str(score))

            elif metric == 'log':
                from sklearn.metrics import log_loss
                score = log_loss(y_test, y_pred)
                scores[metric] = score

                if output == '':
                    output = 'Log Loss: {}'.format(str(score))
                else:
                    output += '\nLog Loss: {}'.format(str(score))

            elif metric == 'mcc':
                from sklearn.metrics import matthews_corrcoef
                score = matthews_corrcoef(y_test, y_pred)
                scores[metric] = score

                if output == '':
                    output = 'MCC: {}'.format(str(score))
                else:
                    output += '\nMCC: {}'.format(str(score))

            elif metric == 'precision':
                from sklearn.metrics import precision_score
                score = precision_score(y_test, y_pred, average=average)
                scores[metric] = score

                if output == '':
                    output = 'Precision Score: {}'.format(str(score))
                else:
                    output += '\nPrecision Score: {}'.format(str(score))

            elif metric == 'recall':
                from sklearn.metrics import recall_score
                score = recall_score(y_test, y_pred, average=average)
                scores[metric] = score

                if output == '':
                    output = 'Recall Score: {}'.format(str(score))
                else:
                    output += '\nRecall Score: {}'.format(str(score))

            elif metric == 'zol':
                from sklearn.metrics import zero_one_loss
                score = zero_one_loss(y_test, y_pred)
                scores[metric] = score

                if output == '':
                    output = 'Zero One Loss: {}'.format(str(score))
                else:
                    output += '\nZero One Loss: {}'.format(str(score))

    elif algo_type == 'reg':
        for metric in metrics:
            if metric == 'var':
                from sklearn.metrics import explained_variance_score
                score = explained_variance_score(y_test, y_pred)
                scores[metric] = score

                if output == '':
                    output = 'Explained Variance Score: {}'.format(str(score))
                else:
                    output += '\nExplained Variance Score: {}'.format(str(score))

            elif metric == 'max':
                from sklearn.metrics import max_error
                score = max_error(y_test, y_pred)
                scores[metric] = score

                if output == '':
                    output = 'Max Error: {}'.format(str(score))
                else:
                    output += '\nMax Error: {}'.format(str(score))

            elif metric == 'abs':
                from sklearn.metrics import mean_absolute_error
                score = mean_absolute_error(y_test, y_pred)
                scores[metric] = score

                if output == '':
                    output = 'Mean Absolute Error: {}'.format(str(score))
                else:
                    output += '\nMean Absolute Error: {}'.format(str(score))

            elif metric == 'sq':
                from sklearn.metrics import mean_squared_error
                score = mean_squared_error(y_test, y_pred)
                scores[metric] = score

                if output == '':
                    output = 'Mean Squared Error: {}'.format(str(score))
                else:
                    output += '\nMean Squared Error: {}'.format(str(score))

            elif metric == 'rsq':
                from sklearn.metrics import root_mean_squared_error
                score = root_mean_squared_error(y_test, y_pred)
                scores[metric] = score

                if output == '':
                    output = 'Root Mean Squared Error: {}'.format(str(score))
                else:
                    output += '\nRoot Mean Squared Error: {}'.format(str(score))

            elif metric == 'log':
                from sklearn.metrics import mean_squared_log_error
                score = mean_squared_log_error(y_test, y_pred)
                scores[metric] = score

                if output == '':
                    output = 'Mean Squared Log Error: {}'.format(str(score))
                else:
                    output += '\nMean Squared Log Error: {}'.format(str(score))

            elif metric == 'rlog':
                from sklearn.metrics import root_mean_squared_log_error
                score = root_mean_squared_log_error(y_test, y_pred)
                scores[metric] = score

                if output == '':
                    output = 'Root Mean Squared Log Error: {}'.format(str(score))
                else:
                    output += '\nRoot Mean Squared Log Error: {}'.format(str(score))

            elif metric == 'medabs':
                from sklearn.metrics import median_absolute_error
                score = median_absolute_error(y_test, y_pred)
                scores[metric] = score

                if output == '':
                    output = 'Median Absolute Error: {}'.format(str(score))
                else:
                    output += '\nMedian Absolute Error: {}'.format(str(score))

            elif metric == 'poisson':
                from sklearn.metrics import mean_poisson_deviance
                score = mean_poisson_deviance(y_test, y_pred)
                scores[metric] = score

                if output == '':
                    output = 'Mean Poisson Deviance: {}'.format(str(score))
                else:
                    output += '\nMean Poisson Deviance: {}'.format(str(score))

            elif metric == 'gamma':
                from sklearn.metrics import mean_gamma_deviance
                score = mean_gamma_deviance(y_test, y_pred)
                scores[metric] = score

                if output == '':
                    output = 'Mean Gamma Deviance: {}'.format(str(score))
                else:
                    output += '\nMean Gamma Deviance: {}'.format(str(score))

            elif metric == 'per':
                from sklearn.metrics import mean_absolute_percentage_error
                score = mean_absolute_percentage_error(y_test, y_pred)
                scores[metric] = score

                if output == '':
                    output = 'Mean Absolute Percentage Error: {}'.format(str(score))
                else:
                    output += '\nMean Absolute Percentage Error: {}'.format(str(score))

            elif metric == 'd2abs':
                from sklearn.metrics import d2_absolute_error_score
                score = d2_absolute_error_score(y_test, y_pred)
                scores[metric] = score

                if output == '':
                    output = 'D2 Absolute Error Score: {}'.format(str(score))
                else:
                    output += '\nD2 Absolute Error Score: {}'.format(str(score))

            elif metric == 'd2pin':
                from sklearn.metrics import d2_pinball_score
                score = d2_pinball_score(y_test, y_pred)
                scores[metric] = score

                if output == '':
                    output = 'D2 Pinball Score: {}'.format(str(score))
                else:
                    output += '\nD2 Pinball Score: {}'.format(str(score))

            elif metric == 'd2twe':
                from sklearn.metrics import d2_tweedie_score
                score = d2_tweedie_score(y_test, y_pred)
                scores[metric] = score

                if output == '':
                    output = 'D2 Tweedie Score: {}'.format(str(score))
                else:
                    output += '\nD2 Tweedie Score: {}'.format(str(score))

    if verbose is True:
        print(output)

    return scores


def do_voting(y_pred_list, combinations, strategy='avg'):
    if strategy == 'avg':
        results = []

        for comb in combinations:
            y_sum = None

            for index in comb:
                if y_sum is None:
                    y_sum = y_pred_list[index]
                else:
                    y_sum += y_pred_list[index]

            y_sum = y_sum / len(comb)
            y_sum = y_sum.astype(int)
            results.append(y_sum)

        return results

    elif strategy == 'mode':
        from math import ceil
        import numpy as np

        results = []

        for comb in combinations:
            selected = []
            for index in comb:
                selected.append(y_pred_list[index])

            stack = np.concatenate(selected, axis=1)
            del selected

            req_min = ceil(len(list(stack[0])) / 2)
            length = stack.shape[0]
            modes = []

            for i in range(length):
                result = 0
                max_times = 0
                min_space = max_times + 1
                row = list(stack[i])

                while len(row) > 0:
                    loc = 0
                    obj = row[0]
                    times = 0

                    while loc < len(row):
                        if obj == row[loc]:
                            del row[loc]
                            times += 1
                        else:
                            loc += 1

                    if times > max_times:
                        result = obj
                        max_times = times
                        min_space = max_times + 1

                    if max_times >= req_min or len(row) < min_space:
                        break

                modes.append(result)

            modes = np.array(modes)
            results.append(modes)

        return results


def do_combinations(indexes, min_item, max_item):
    import itertools
    combinations = []

    for i in range(min_item, max_item + 1):
        combs = list(itertools.combinations(indexes, i))

        for comb in combs:
            combinations.append(list(comb))

    return combinations


def examine_time(model, X_train, y_train):
    import time

    start = time.process_time()
    model.fit(X_train, y_train)
    end = time.process_time()

    consumed = end - start
    return consumed, model


class WelkinClassification:
    def __init__(self, strategy='travel', priority=None, limit=None):
        self.min = {}
        self.max = {}
        self.strategy = strategy
        self.priority = priority
        self.limit = limit

    def fit(self, X_train, y_train):
        import sys

        for i in range(y_train.shape[0]):
            if not y_train[i] in self.min:
                self.min[y_train[i]] = {}
                self.max[y_train[i]] = {}

                for j in range(X_train.shape[1]):
                    self.min[y_train[i]][j] = sys.maxsize
                    self.max[y_train[i]][j] = -sys.maxsize - 1

            row = list(X_train[i])

            for j in range(len(row)):
                if row[j] < self.min[y_train[i]][j]:
                    self.min[y_train[i]][j] = row[j]

                elif row[j] > self.max[y_train[i]][j]:
                    self.max[y_train[i]][j] = row[j]

    def predict(self, X_test):
        if self.strategy == 'travel':
            import numpy as np
            from random import randint

            pred = []

            stage_line = list(self.min.keys())
            if self.priority is not None:
                stage_line = self.priority

            for i in range(X_test.shape[0]):
                max_key = list(self.max.keys())[randint(0, len(list(self.max.keys())) - 1)]
                max_zone = 0

                row = list(X_test[i])

                for output in stage_line:
                    count = 0

                    for j in range(X_test.shape[1]):
                        if row[j] >= self.min[output][j] and row[j] <= self.max[output][j]:
                            count += 1

                    if count > max_zone:
                        max_zone = count
                        max_key = output

                        if count == X_test.shape[0]:
                            break

                pred.append(max_key)

            return np.array(pred)

        elif self.strategy == 'limit' and self.limit is not None:
            import numpy as np
            from random import randint

            pred = []

            stage_line = list(self.min.keys())
            if self.priority is not None:
                stage_line = self.priority

            for i in range(X_test.shape[0]):
                max_key = list(self.max.keys())[randint(0, len(list(self.max.keys())) - 1)]
                max_zone = 0

                row = list(X_test[i])

                for output in stage_line:
                    count = 0

                    for j in range(X_test.shape[1]):
                        if row[j] >= self.min[output][j] and row[j] <= self.max[output][j]:
                            count += 1

                    if count > max_zone:
                        max_zone = count
                        max_key = output

                        if count == X_test.shape[0] or count >= self.limit:
                            break

                pred.append(max_key)

            return np.array(pred)


class DistRegressor:
    import numpy as np

    def __init__(self, verbose=True, clf_model=None, clf_params=None, reg_model=None, reg_params=None,
                 efficiency='time', rus=True):
        self.type_zero_regressor = None
        self.type_one_regressor = None
        self.type_two_regressor = None
        self.rus = rus
        self.verbose = verbose
        self.efficiency = efficiency

        self.clf_model = clf_model
        self.reg_model = reg_model
        self.reg_params = reg_params

        if self.clf_model is not None:
            if clf_params is None:
                self.clf_model = clf_model()
            else:
                self.clf_model = clf_model(**clf_params)

    def fit(self, X_train, y_train):
        std = np.std(y_train)
        mean = np.mean(y_train)
        amin = np.amin(y_train)
        amax = np.amax(y_train)

        border_one = mean - std
        border_two = mean + std

        if amin >= border_one or amax <= border_two:
            raise ValueError('There is no such a normal distribution!')

        if self.verbose:
            print('Basic calculations are completed')

        clf_arr = []
        one_side = 0
        type_zero_X = []
        type_one_X = []
        type_two_X = []
        type_zero_y = []
        type_one_y = []
        type_two_y = []

        if self.efficiency == 'space':
            for i in range(y_train.shape[0]):
                if y_train[i] <= border_one:
                    clf_arr.append(0)
                    one_side += 1
                elif y_train[i] > border_one and y_train[i] < border_two:
                    clf_arr.append(1)
                else:
                    clf_arr.append(2)

            del type_zero_X, type_one_X, type_two_X, type_zero_y, type_one_y, type_two_y

        elif self.efficiency == 'time':
            for i in range(y_train.shape[0]):
                if y_train[i] <= border_one:
                    clf_arr.append(0)
                    type_zero_X.append(list(X_train[i]))
                    type_zero_y.append(y_train[i])
                elif y_train[i] > border_one and y_train[i] < border_two:
                    clf_arr.append(1)
                    type_one_X.append(list(X_train[i]))
                    type_one_y.append(y_train[i])
                else:
                    clf_arr.append(2)
                    type_two_X.append(list(X_train[i]))
                    type_two_y.append(y_train[i])

            type_zero_X = np.array(type_zero_X)
            type_one_X = np.array(type_one_X)
            type_two_X = np.array(type_two_X)
            type_zero_y = np.array(type_zero_y)
            type_one_y = np.array(type_one_y)
            type_two_y = np.array(type_two_y)

        clf_arr = np.array(clf_arr)

        if self.verbose:
            print('Result array is ready for classification')

        small_X = None
        if self.rus:
            from imblearn.under_sampling import RandomUnderSampler

            strategy = {1: one_side}
            rand = RandomUnderSampler(sampling_strategy=strategy)

            small_X, clf_arr = rand.fit_resample(X_train, clf_arr)

        if self.clf_model is None:
            from catboost import CatBoostClassifier

            self.clf_model = CatBoostClassifier(verbose=False, iterations=20)

            if self.rus:
                self.clf_model.fit(small_X, clf_arr)
            else:
                self.clf_model.fit(X_train, clf_arr)
        else:
            if self.rus:
                self.clf_model.fit(small_X, clf_arr)
            else:
                self.clf_model.fit(X_train, clf_arr)

        del small_X, clf_arr

        if self.verbose:
            print('Classification model has been trained')

        if self.efficiency == 'space':
            sub_X = []
            sub_y = []
            for i in range(X_train.shape[0]):
                if y_train[i] <= border_one:
                    sub_X.append(list(X_train[i]))
                    sub_y.append(y_train[i])

            sub_X = np.array(sub_X)
            sub_y = np.array(sub_y)

            if self.reg_model is None:
                from catboost import CatBoostRegressor

                self.type_zero_regressor = CatBoostRegressor(verbose=False, iterations=20)
                self.type_zero_regressor.fit(sub_X, sub_y)
            else:
                if self.reg_params is None:
                    self.type_zero_regressor = self.reg_model()
                else:
                    self.type_zero_regressor = self.reg_model(**self.reg_params)

                self.type_zero_regressor.fit(sub_X, sub_y)

            sub_X = []
            sub_y = []
            for i in range(X_train.shape[0]):
                if y_train[i] > border_one and y_train[i] < border_two:
                    sub_X.append(list(X_train[i]))
                    sub_y.append(y_train[i])

            sub_X = np.array(sub_X)
            sub_y = np.array(sub_y)

            if self.reg_model is None:
                from catboost import CatBoostRegressor

                self.type_one_regressor = CatBoostRegressor(verbose=False, iterations=20)
                self.type_one_regressor.fit(sub_X, sub_y)
            else:
                if self.reg_params is None:
                    self.type_one_regressor = self.reg_model()
                else:
                    self.type_one_regressor = self.reg_model(**self.reg_params)

                self.type_one_regressor.fit(sub_X, sub_y)

            sub_X = []
            sub_y = []
            for i in range(X_train.shape[0]):
                if y_train[i] >= border_two:
                    sub_X.append(list(X_train[i]))
                    sub_y.append(y_train[i])

            sub_X = np.array(sub_X)
            sub_y = np.array(sub_y)

            if self.reg_model is None:
                from catboost import CatBoostRegressor

                self.type_two_regressor = CatBoostRegressor(verbose=False, iterations=20)
                self.type_two_regressor.fit(sub_X, sub_y)
            else:
                if self.reg_params is None:
                    self.type_two_regressor = self.reg_model()
                else:
                    self.type_two_regressor = self.reg_model(**self.reg_params)

                self.type_two_regressor.fit(sub_X, sub_y)

        elif self.efficiency == 'time':
            if self.reg_model is None:
                from catboost import CatBoostRegressor

                self.type_zero_regressor = CatBoostRegressor(verbose=False, iterations=20)
                self.type_one_regressor = CatBoostRegressor(verbose=False, iterations=20)
                self.type_two_regressor = CatBoostRegressor(verbose=False, iterations=20)

                self.type_zero_regressor.fit(type_zero_X, type_zero_y)
                del type_zero_X, type_zero_y

                self.type_one_regressor.fit(type_one_X, type_one_y)
                del type_one_X, type_one_y

                self.type_two_regressor.fit(type_two_X, type_two_y)
                del type_two_X, type_two_y
            else:
                if self.reg_params is None:
                    self.type_zero_regressor = self.reg_model()
                    self.type_one_regressor = self.reg_model()
                    self.type_two_regressor = self.reg_model()
                else:
                    self.type_zero_regressor = self.reg_model(**self.reg_params)
                    self.type_one_regressor = self.reg_model(**self.reg_params)
                    self.type_two_regressor = self.reg_model(**self.reg_params)

                self.type_zero_regressor.fit(type_zero_X, type_zero_y)
                del type_zero_X, type_zero_y

                self.type_one_regressor.fit(type_one_X, type_one_y)
                del type_one_X, type_one_y

                self.type_two_regressor.fit(type_two_X, type_two_y)
                del type_two_X, type_two_y

        if self.verbose:
            print('Regression models have been trained')

    def predict(self, X_test):
        y_pred = []

        for i in range(X_test.shape[0]):
            category = self.clf_model.predict([X_test[i]])

            if category == 0:
                y_pred.append(self.type_zero_regressor.predict([X_test[i]]))
            elif category == 1:
                y_pred.append(self.type_one_regressor.predict([X_test[i]]))
            else:
                y_pred.append(self.type_two_regressor.predict([X_test[i]]))

        y_pred = np.array(y_pred)
        return y_pred

    def is_data_normal(self, y):
        amax = np.amax(y)
        amin = np.amin(y)
        mean = np.mean(y)
        std = np.std(y)

        if amin < mean - std and amax > mean + std:
            return True
        else:
            return False


def compare_models(algo_type, algorithms, metrics, X_train, y_train, X_test, y_test, get_result=False):
    results = {}

    if algo_type == 'clf':
        from catboost import CatBoostClassifier
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from lightgbm import LGBMClassifier
        from sklearn.tree import ExtraTreeClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.linear_model import RidgeClassifier
        from sklearn.naive_bayes import BernoulliNB
        from sklearn.svm import SVC
        from sklearn.linear_model import Perceptron
        from sklearn.naive_bayes import MultinomialNB

        if algorithms[0] == 'all':
            algorithms = ['cat', 'ada', 'dtr', 'raf', 'lbm', 'ext', 'log', 'knn', 'gnb', 'rdg', 'bnb', 'svc', 'per', 'mnb']

        for algo in algorithms:
            if algo == 'cat':
                model = CatBoostClassifier(verbose=False)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('CatBoost')
                results[algo] = get_score(y_test, y_pred, metrics)
                print('***')

            elif algo == 'ada':
                model = AdaBoostClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('AdaBoost')
                results[algo] = get_score(y_test, y_pred, metrics)
                print('***')

            elif algo == 'dtr':
                model = DecisionTreeClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('Decision Tree')
                results[algo] = get_score(y_test, y_pred, metrics)
                print('***')

            elif algo == 'raf':
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('Random Forest')
                results[algo] = get_score(y_test, y_pred, metrics)
                print('***')

            elif algo == 'lbm':
                model = LGBMClassifier(verbosity=-1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('LightGBM')
                results[algo] = get_score(y_test, y_pred, metrics)
                print('***')

            elif algo == 'ext':
                model = ExtraTreeClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('Extra Tree')
                results[algo] = get_score(y_test, y_pred, metrics)
                print('***')

            elif algo == 'log':
                model = LogisticRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('Logistic Regression')
                results[algo] = get_score(y_test, y_pred, metrics)
                print('***')

            elif algo == 'knn':
                model = KNeighborsClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('KNN')
                results[algo] = get_score(y_test, y_pred, metrics)
                print('***')

            elif algo == 'gnb':
                model = GaussianNB()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('GaussianNB')
                results[algo] = get_score(y_test, y_pred, metrics)
                print('***')

            elif algo == 'rdg':
                model = RidgeClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('Ridge')
                results[algo] = get_score(y_test, y_pred, metrics)
                print('***')

            elif algo == 'bnb':
                model = BernoulliNB()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('BernoulliNB')
                results[algo] = get_score(y_test, y_pred, metrics)
                print('***')

            elif algo == 'svc':
                model = SVC()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('SVC')
                results[algo] = get_score(y_test, y_pred, metrics)
                print('***')

            elif algo == 'per':
                model = Perceptron()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('Perceptron')
                results[algo] = get_score(y_test, y_pred, metrics)
                print('***')

            elif algo == 'mnb':
                model = MultinomialNB()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('MultinomialNB')
                results[algo] = get_score(y_test, y_pred, metrics)
                print('***')

    elif algo_type == 'reg':
        from catboost import CatBoostRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from lightgbm import LGBMRegressor
        from sklearn.tree import ExtraTreeRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVR

        if algorithms[0] == 'all':
            algorithms = ['cat', 'ada', 'dtr', 'raf', 'lbm', 'ext', 'lin', 'knn', 'svr']

        for algo in algorithms:
            if algo == 'cat':
                model = CatBoostRegressor(verbose=False)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('CatBoost')
                results[algo] = get_score(y_test, y_pred, metrics, algo_type='reg')
                print('***')

            elif algo == 'ada':
                model = AdaBoostRegressor()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('AdaBoost')
                results[algo] = get_score(y_test, y_pred, metrics, algo_type='reg')
                print('***')

            elif algo == 'dtr':
                model = DecisionTreeRegressor()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('Decision Tree')
                results[algo] = get_score(y_test, y_pred, metrics, algo_type='reg')
                print('***')

            elif algo == 'raf':
                model = RandomForestRegressor()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('Random Forest')
                results[algo] = get_score(y_test, y_pred, metrics, algo_type='reg')
                print('***')

            elif algo == 'lbm':
                model = LGBMRegressor(verbosity=-1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('LightGBM')
                results[algo] = get_score(y_test, y_pred, metrics, algo_type='reg')
                print('***')

            elif algo == 'ext':
                model = ExtraTreeRegressor()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('Extra Tree')
                results[algo] = get_score(y_test, y_pred, metrics, algo_type='reg')
                print('***')

            elif algo == 'lin':
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('Linear Regression')
                results[algo] = get_score(y_test, y_pred, metrics, algo_type='reg')
                print('***')

            elif algo == 'knn':
                model = KNeighborsRegressor()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('KNN')
                results[algo] = get_score(y_test, y_pred, metrics, algo_type='reg')
                print('***')

            elif algo == 'svr':
                model = SVR()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                print('SVR')
                results[algo] = get_score(y_test, y_pred, metrics, algo_type='reg')
                print('***')

    if get_result is True:
        return results


def get_best_model(scores, rel_metric, algo_type, X_train, y_train, behavior='min-best', verbose=True):
    model = None
    fit_algo = None
    fit_metrics = None

    if behavior == 'min-best':
        for algo in scores:
            if fit_metrics is None:
                fit_algo = algo
                fit_metrics = scores[algo][rel_metric]
            elif scores[algo][rel_metric] < fit_metrics:
                fit_algo = algo
                fit_metrics = scores[algo][rel_metric]

    elif behavior == 'max-best':
        for algo in scores:
            if fit_metrics is None:
                fit_algo = algo
                fit_metrics = scores[algo][rel_metric]
            elif scores[algo][rel_metric] > fit_metrics:
                fit_algo = algo
                fit_metrics = scores[algo][rel_metric]

    if algo_type == 'clf':
        from catboost import CatBoostClassifier
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from lightgbm import LGBMClassifier
        from sklearn.tree import ExtraTreeClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.linear_model import RidgeClassifier
        from sklearn.naive_bayes import BernoulliNB
        from sklearn.svm import SVC
        from sklearn.linear_model import Perceptron
        from sklearn.naive_bayes import MultinomialNB

        if fit_algo == 'cat':
            model = CatBoostClassifier(verbose=False)
            model.fit(X_train, y_train)

        elif fit_algo == 'ada':
            model = AdaBoostClassifier()
            model.fit(X_train, y_train)

        elif fit_algo == 'dtr':
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)

        elif fit_algo == 'raf':
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

        elif fit_algo == 'lbm':
            model = LGBMClassifier(verbosity=-1)
            model.fit(X_train, y_train)

        elif fit_algo == 'ext':
            model = ExtraTreeClassifier()
            model.fit(X_train, y_train)

        elif fit_algo == 'log':
            model = LogisticRegression()
            model.fit(X_train, y_train)

        elif fit_algo == 'knn':
            model = KNeighborsClassifier()
            model.fit(X_train, y_train)

        elif fit_algo == 'gnb':
            model = GaussianNB()
            model.fit(X_train, y_train)

        elif fit_algo == 'rdg':
            model = RidgeClassifier()
            model.fit(X_train, y_train)

        elif fit_algo == 'bnb':
            model = BernoulliNB()
            model.fit(X_train, y_train)

        elif fit_algo == 'svc':
            model = SVC()
            model.fit(X_train, y_train)

        elif fit_algo == 'per':
            model = Perceptron()
            model.fit(X_train, y_train)

        elif fit_algo == 'mnb':
            model = MultinomialNB()
            model.fit(X_train, y_train)

    elif algo_type == 'reg':
        from catboost import CatBoostRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from lightgbm import LGBMRegressor
        from sklearn.tree import ExtraTreeRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVR

        if fit_algo == 'cat':
            model = CatBoostRegressor(verbose=False)
            model.fit(X_train, y_train)

        elif fit_algo == 'ada':
            model = AdaBoostRegressor()
            model.fit(X_train, y_train)

        elif fit_algo == 'dtr':
            model = DecisionTreeRegressor()
            model.fit(X_train, y_train)

        elif fit_algo == 'raf':
            model = RandomForestRegressor()
            model.fit(X_train, y_train)

        elif fit_algo == 'lbm':
            model = LGBMRegressor(verbosity=-1)
            model.fit(X_train, y_train)

        elif fit_algo == 'ext':
            model = ExtraTreeRegressor()
            model.fit(X_train, y_train)

        elif fit_algo == 'lin':
            model = LinearRegression()
            model.fit(X_train, y_train)

        elif fit_algo == 'knn':
            model = KNeighborsRegressor()
            model.fit(X_train, y_train)

        elif fit_algo == 'svr':
            model = SVR()
            model.fit(X_train, y_train)

    if verbose is True:
        print('Best Algorithm is {} with the score of {}'.format(fit_algo, fit_metrics))

    return model


def subacc(y_test, y_pred, get_general=False):
    from sklearn.metrics import accuracy_score as accs

    y_test = list(y_test)
    y_pred = list(y_pred)

    population = {}
    succeeded = {}

    uniqs = np.unique(y_test)

    for uniq in uniqs:
        population[uniq] = 0
        succeeded[uniq] = 0

    for i in range(len(y_test)):
        population[y_test[i]] += 1

        if y_test[i] == y_pred[i]:
            succeeded[y_test[i]] += 1

    acc = {}

    for uniq in uniqs:
        acc[uniq] = succeeded[uniq] / population[uniq]

    if get_general is True:
        score = accs(y_test, y_pred)
        return acc, score
    else:
        return acc


def get_models(algorithms, X_train, y_train, algo_type='clf'):
    if algo_type == 'clf':
        from catboost import CatBoostClassifier
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from lightgbm import LGBMClassifier
        from sklearn.tree import ExtraTreeClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.linear_model import RidgeClassifier
        from sklearn.naive_bayes import BernoulliNB
        from sklearn.svm import SVC
        from sklearn.linear_model import Perceptron
        from sklearn.naive_bayes import MultinomialNB

        models = {}

        if algorithms[0] == 'all':
            algorithms = ['cat', 'ada', 'dtr', 'raf', 'lbm', 'ext', 'log', 'knn', 'gnb', 'rdg', 'bnb', 'svc', 'per', 'mnb']

        for algo in algorithms:
            if algo == 'cat':
                model = CatBoostClassifier(verbose=False)
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'ada':
                model = AdaBoostClassifier()
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'dtr':
                model = DecisionTreeClassifier()
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'raf':
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'lbm':
                model = LGBMClassifier(verbosity=-1)
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'ext':
                model = ExtraTreeClassifier()
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'log':
                model = LogisticRegression()
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'knn':
                model = KNeighborsClassifier()
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'gnb':
                model = GaussianNB()
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'rdg':
                model = RidgeClassifier()
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'bnb':
                model = BernoulliNB()
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'svc':
                model = SVC()
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'per':
                model = Perceptron()
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'mnb':
                model = MultinomialNB()
                model.fit(X_train, y_train)
                models[algo] = model

        return models

    elif algo_type == 'reg':
        from catboost import CatBoostRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from lightgbm import LGBMRegressor
        from sklearn.tree import ExtraTreeRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVR

        models = {}

        if algorithms[0] == 'all':
            algorithms = ['cat', 'ada', 'dtr', 'raf', 'lbm', 'ext', 'lin', 'knn', 'svr']

        for algo in algorithms:
            if algo == 'cat':
                model = CatBoostRegressor(verbose=False)
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'ada':
                model = AdaBoostRegressor()
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'dtr':
                model = DecisionTreeRegressor()
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'raf':
                model = RandomForestRegressor()
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'lbm':
                model = LGBMRegressor(verbosity=-1)
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'ext':
                model = ExtraTreeRegressor()
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'lin':
                model = LinearRegression()
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'knn':
                model = KNeighborsRegressor()
                model.fit(X_train, y_train)
                models[algo] = model

            elif algo == 'svr':
                model = SVR()
                model.fit(X_train, y_train)
                models[algo] = model

        return models


def commune_create(algorithms, X_train, y_train, X_val, y_val, get_dict=False):
    from sklearn.metrics import accuracy_score

    models = get_models(algorithms, X_train, y_train)
    results = {}

    for model in models:
        y_pred = models[model].predict(X_val)
        sub, gen = subacc(y_val, y_pred, get_general=True)

        results[model] = {'gen': gen,
                          'sub': sub}

    uniqs = np.unique(y_val)

    gen_best = ''
    gen_best_score = 0
    bests = {}

    for uniq in uniqs:
        bests[uniq] = {'score': 0, 'algo': ''}

    for algo in results:
        if results[algo]['gen'] > gen_best_score:
            gen_best = str(algo)
            gen_best_score = results[algo]['gen']

    del results[gen_best]

    for algo in results:
        for uniq in uniqs:
            if results[algo]['sub'][uniq] > bests[uniq]['score']:
                bests[uniq]['algo'] = str(algo)
                bests[uniq]['score'] = results[algo]['sub'][uniq]

    y_pred = []
    recatch = []
    y_re = []

    instead = ''
    instead_score = 0

    for i in range(X_val.shape[0]):
        res = models[gen_best].predict([X_val[i, :]])[0]
        check = models[bests[res]['algo']].predict([X_val[i, :]])[0]

        if res == check:
            y_pred.append(res)
        else:
            y_pred.append(None)
            recatch.append(X_val[i, :])
            y_re.append(y_val[i])

    for algo in models:
        y_sub = models[algo].predict(recatch)
        sco = accuracy_score(y_re, y_sub)

        if sco > instead_score:
            instead = algo
            instead_score = sco

    y_sub = models[instead].predict(recatch)
    loc = 0

    for i in range(len(y_pred)):
        if y_pred[i] is None:
            y_pred[i] = y_sub[loc]
            loc += 1

    if get_dict is True:
        if instead == '':
            instead = gen_best

        fin = {
            'models': models,
            'general': gen_best,
            'best': bests,
            'instead': instead
        }

        return y_pred, fin
    else:
        return y_pred


def commune_apply(declaration, X_test):
    y_pred = []

    for i in range(X_test.shape[0]):
        res = declaration['models'][declaration['general']].predict([X_test[i, :]])[0]
        check = declaration['models'][declaration['best'][res]['algo']].predict([X_test[i, :]])[0]

        if res == check:
            y_pred.append(res)
        else:
            res = declaration['models'][declaration['instead']].predict([X_test[i, :]])[0]
            y_pred.append(res)

    y_pred = np.array(y_pred)
    return y_pred


def find_deflection(y_test, y_pred, arr=True, avg=False, gap=None, gap_type='num', dif_type='f-i', avg_w_abs=True, success_indexes=False):
    if arr == True or avg == True or gap is not None:
        diffs = []

        first = None
        second = None

        if dif_type == 'f-i':
            first = y_pred
            second = y_test
        elif dif_type == 'i-f':
            first = y_test
            second = y_pred
        else:
            first = y_pred
            second = y_test

        for i in range(y_test.shape[0]):
            diffs.append(first[i] - second[i])

        diffs = np.array(diffs)

        if dif_type == 'abs':
            diffs = np.abs(diffs)

        if arr == True and avg == False and gap is None:
            return diffs

        if avg == True:
            avg_score = 0

            tot_arr = diffs
            if dif_type != 'abs' and avg_w_abs == True:
                tot_arr = np.abs(diffs)

            for i in range(diffs.shape[0]):
                avg_score += tot_arr[i]

            avg_score /= diffs.shape[0]

            if arr == True and avg == True and gap is None:
                return diffs, avg_score

            if arr == False and avg == True and gap is None:
                return avg_score

            indexes = []

            if gap_type == 'exact':
                for i in range(y_test.shape[0]):
                    if y_test[i] == y_pred[i]:
                        indexes.append(i)
            else:
                for i in range(y_test.shape[0]):
                    lowest = None
                    highest = None

                    if gap_type == 'num':
                        lowest = y_test[i] - gap
                        highest = y_test[i] + gap
                    elif gap_type == 'num+':
                        lowest = y_test[i]
                        highest = y_test[i] + gap
                    elif gap_type == 'num-':
                        lowest = y_test[i] - gap
                        highest = y_test[i]
                    elif gap_type == 'per':
                        lowest = y_test[i] * (100 - gap) / 100
                        highest = y_test[i] * (100 + gap) / 100
                    elif gap_type == 'per+':
                        lowest = y_test[i]
                        highest = y_test[i] * (100 + gap) / 100
                    elif gap_type == 'per-':
                        lowest = y_test[i] * (100 - gap) / 100
                        highest = y_test[i]

                    if highest < lowest:
                        temp = lowest
                        lowest = highest
                        highest = temp

                    if lowest <= y_pred[i] <= highest:
                        indexes.append(i)

            if arr == True and avg == True and gap is not None:
                if success_indexes == True:
                    return diffs, avg_score, len(indexes), indexes
                else:
                    return diffs, avg_score, len(indexes)

            if arr == False and avg == True and gap is not None:
                if success_indexes == True:
                    return avg_score, len(indexes), indexes
                else:
                    return avg_score, len(indexes)

        if gap is not None:
            indexes = []

            if gap_type == 'exact':
                for i in range(y_test.shape[0]):
                    if y_test[i] == y_pred[i]:
                        indexes.append(i)
            else:
                for i in range(y_test.shape[0]):
                    lowest = None
                    highest = None

                    if gap_type == 'num':
                        lowest = y_test[i] - gap
                        highest = y_test[i] + gap
                    elif gap_type == 'num+':
                        lowest = y_test[i]
                        highest = y_test[i] + gap
                    elif gap_type == 'num-':
                        lowest = y_test[i] - gap
                        highest = y_test[i]
                    elif gap_type == 'per':
                        lowest = y_test[i] * (100 - gap) / 100
                        highest = y_test[i] * (100 + gap) / 100
                    elif gap_type == 'per+':
                        lowest = y_test[i]
                        highest = y_test[i] * (100 + gap) / 100
                    elif gap_type == 'per-':
                        lowest = y_test[i] * (100 - gap) / 100
                        highest = y_test[i]

                    if highest < lowest:
                        temp = lowest
                        lowest = highest
                        highest = temp

                    if lowest <= y_pred[i] <= highest:
                        indexes.append(i)

            if arr == True and avg == False:
                if success_indexes == True:
                    return diffs, len(indexes), indexes
                else:
                    return diffs, len(indexes)

            if arr == False and avg == False:
                if success_indexes == True:
                    return len(indexes), indexes
                else:
                    return len(indexes)
