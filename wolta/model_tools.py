def get_score(y_test, y_pred, metrics=None, average='weighted'):
    from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score, log_loss, matthews_corrcoef, precision_score, recall_score, zero_one_loss

    if metrics is None:
        metrics = ['acc']

    output = ''
    scores = {}

    for metric in metrics:
        if metric == 'acc':
            score = accuracy_score(y_test, y_pred)
            scores[metric] = score

            if output == '':
                output = 'Accuracy Score: {}'.format(str(score))
            else:
                output += '\nAccuracy Score: {}'.format(str(score))

        elif metric == 'f1':
            score = f1_score(y_test, y_pred, average=average)
            scores[metric] = score

            if output == '':
                output = 'F1 Score (weighted): {}'.format(str(score))
            else:
                output += '\nF1 Score (weighted): {}'.format(str(score))

        elif metric == 'hamming':
            score = hamming_loss(y_test, y_pred)
            scores[metric] = score

            if output == '':
                output = 'Hamming Loss: {}'.format(str(score))
            else:
                output += '\nHamming Loss: {}'.format(str(score))

        elif metric == 'jaccard':
            score = jaccard_score(y_test, y_pred, average=average)
            scores[metric] = score

            if output == '':
                output = 'Jaccard: {}'.format(str(score))
            else:
                output += '\nJaccard: {}'.format(str(score))

        elif metric == 'log':
            score = log_loss(y_test, y_pred)
            scores[metric] = score

            if output == '':
                output = 'Log Loss: {}'.format(str(score))
            else:
                output += '\nLog Loss: {}'.format(str(score))

        elif metric == 'mcc':
            score = matthews_corrcoef(y_test, y_pred)
            scores[metric] = score

            if output == '':
                output = 'MCC: {}'.format(str(score))
            else:
                output += '\nMCC: {}'.format(str(score))

        elif metric == 'precision':
            score = precision_score(y_test, y_pred, average=average)
            scores[metric] = score

            if output == '':
                output = 'Precision Score: {}'.format(str(score))
            else:
                output += '\nPrecision Score: {}'.format(str(score))

        elif metric == 'recall':
            score = recall_score(y_test, y_pred, average=average)
            scores[metric] = score

            if output == '':
                output = 'Recall Score: {}'.format(str(score))
            else:
                output += '\nRecall Score: {}'.format(str(score))

        elif metric == 'zol':
            score = zero_one_loss(y_test, y_pred)
            scores[metric] = score

            if output == '':
                output = 'Zero One Loss: {}'.format(str(score))
            else:
                output += '\nZero One Loss: {}'.format(str(score))

    print(output)
    return scores


def get_supported_metrics():
    return ['acc', 'f1', 'hamming', 'jaccard', 'log', 'mcc', 'precision', 'recall', 'zol']


def get_avg_options():
    return ['micro', 'macro', 'binary', 'weighted', 'samples']


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
