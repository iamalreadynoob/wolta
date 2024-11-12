# Data

In order to test some functions in Wolta, there are some datasets used by test functions. They are located in the test_data folder and this page contains all information about them. One more thing, they are named with human names in English. Why? I do not know, I think it is cool.

## Sasha

### General Information

| specification | respond |
| --- | --- |
| source | created for Wolta |
| location | traditional folder |
| type | multi class classification |
| version | 1.0 |
| samples | 4000 |
| input features | 10 |
| label amount | 1 |
| started | 11/12/2024 |
| finalized | 11/12/2024 |
| added | 11/22/2024 |
| last updated | 11/22/2024|

### Discrete Features

| feature name | unique value amount | unique set |
| --- | --- | --- |
| discrete1 | 7 | 0, 1, 2, 3, 4, 5, 6 |
| discrete2 | 3 | 0, 1, 2 |
| discrete3 | 5 | 0, 1, 2, 3, 4 |

### Continuous Features

| feature name | min value | max value |
| --- | --- | --- |
| continuous1 | 72 | 117 |
| continuous2 | 75 | 1049 |
| continuous3 | 2 | 439 |
| continuous4 | 37 | 148 |
| continuous5 | 20 | 68 |
| continuous6 | 68 | 427 |
| continuous7 | 84 | 858 |

### Classes

| class | amount |
| --- | --- |
| 0 | 1000 |
| 1 | 1000 |
| 2 | 1000 |
| 3 | 1000 |

### Source Code

```python
from random import randint

def header():
    text = ''
    features = []

    for i in range(3):
        ftr = 'discrete{}'.format((i+1))
        text += '{},'.format(ftr)
        features.append(ftr)

    for i in range(7):
        ftr = 'continuous{}'.format((i+1))
        text += '{},'.format(ftr)
        features.append(ftr)

    text += 'output'

    return text, features


def discrete():
    results = {}

    for i in range(3):
        ftr = 'discrete{}'.format((i+1))
        cls_amount = randint(2, 10)

        results[ftr] = cls_amount

    return results


def continuous():
    results = {}

    for i in range(7):
        ftr = 'continuous{}'.format((i+1))
        min_val = randint(1, 100)
        val_range = randint(20, 1000)
        max_val = min_val + val_range

        results[ftr] = {
            'min': min_val,
            'max': max_val
        }

    return results


def add_line(features, c_map, d_map, out):
    line = '\n'

    for ftr in features:
        if ftr in c_map:
            value = str(randint(c_map[ftr]['min'], c_map[ftr]['max']))
        else:
            value = str(randint(0, d_map[ftr]))

        line += '{},'.format(value)

    line += out
    return line


if __name__ == '__main__':
    text, features = header()
    d_map = discrete()
    c_map = continuous()

    for out in ['0', '1', '2', '3']:
        for _ in range(1000):
            text += add_line(features, c_map, d_map, out)

    with open('df-small-fixed.csv', 'w') as f:
        f.write(text)
    
    #for info
    print(d_map)
    print(c_map)
```