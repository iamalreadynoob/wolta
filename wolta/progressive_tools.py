def make_run(model_class, X_train, y_train, X_test, y_test, init_per=1, limit_per=100, increment=1, metrics=None, average='weighted', params=None):
    from model_tools import get_score

    if metrics is None:
        metrics = ['acc']

    percentage_log = []
    metrics_log = []

    for i in range(init_per, limit_per + 1, increment):
        chunksize = X_train.shape[0] * i // 100

        if params is not None:
            model = model_class(**params)
        else:
            model = model_class()

        model.fit(X_train[:chunksize], y_train[:chunksize])
        y_pred = model.predict(X_test)
        scores = get_score(y_test, y_pred, metrics=metrics, average=average)

        percentage_log.append(i)
        metrics_log.append(scores)

    return percentage_log, metrics_log


def get_best(percentage_log, metrics_log, requested_metrics):
    best_percentage = -1
    best_score = 0

    for i in range(len(percentage_log)):
        if metrics_log[i][requested_metrics] > best_score:
            best_percentage = percentage_log[i]
            best_score = metrics_log[i][requested_metrics]

    print('best score for {} is {} with {}% of train data'.format(requested_metrics, str(best_score), str(best_percentage)))

    return best_percentage, best_score

def path_chain(paths, model_class, X_test, y_test, output_column, metrics=None, average='weighted', params=None):
    import pandas as pd
    from model_tools import get_score

    if metrics is None:
        metrics = ['acc']

    metrics_log = []

    for path in paths:
        df = pd.read_csv(path)
        y_train = df[output_column].values
        del df[output_column]
        X_train = df.values
        del df

        if params is not None:
            model = model_class(**params)
        else:
            model = model_class()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics_log.append(get_score(y_test, y_pred, metrics=metrics, average=average))

    return metrics_log