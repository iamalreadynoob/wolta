def make_table(mode, inputs=None, inputs_type='row', first_column=None, filler=None):
    result = ''
    if mode == 'sheet':
        if inputs_type == 'row':
            if len(inputs) > 0:
                next_ln = '|'
                result = '|'
                for i in range(len(inputs[0])):
                    result += ' ' + inputs[0][i] + ' |'
                    next_ln += ' --- |'

                result += '\n' + next_ln

                for i in range(1, len(inputs)):
                    next_ln = '|'

                    for j in range(len(inputs[i])):
                        next_ln += ' ' + inputs[i][j] + ' |'

                    result += '\n' + next_ln

        elif inputs_type == 'column':
            if len(inputs) > 0:
                next_ln = '|'
                result = '|'

                for i in range(len(inputs)):
                    result += ' ' + inputs[i][0] + ' |'
                    next_ln += ' --- |'

                result += '\n' + next_ln

                for i in range(1, len(inputs[0])):
                    next_ln = '|'

                    for j in range(len(inputs)):
                        next_ln += ' ' + inputs[j][i] + ' |'

                    result += '\n' + next_ln

    elif mode == 'nx2':
        if len(first_column) > 0:
            result = '| {} | {} |\n| --- | --- |'.format(first_column[0], filler)

            for i in range(1, len(first_column)):
                result += '\n| ' + first_column[i] + ' | ' + filler + ' |'

    return result
