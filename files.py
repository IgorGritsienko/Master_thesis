import os
import pandas as pd
import numpy as np

def Read_parameters(filename):
    parameters = []
    with open(filename, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parameters.append(line)
    parameters = [x.rstrip("\n") for x in parameters]                          # удалить '/n' символы из элементов массива
    return parameters

# sep=',' для записи в один столбец через ','
def Read_csv(filename, n_rows, position):
    df = pd.read_csv(filename, nrows=n_rows, skiprows=position, sep=';', header=0)
    return df

def Save_sample_to_csv(array, filename, _mode):
    df = pd.DataFrame(data={"Weeks": array[:, 0], "CTR": array[:, 1], "target": array[:, 2]})       # 3 столбца датафрейма: КТИ, недели и класс (анемия или норма)
    df.to_csv(filename, sep=';',index=False, mode=_mode, header=False, float_format='%.6f')              # mode 'a' - добавление новых записей к старым, создание файла при его отсутствии. 'w' - перезапись


def quality_criteria_results_print_save_to_file(spec, sens, roc, prob_norm, prob_anem):
    np.savetxt('output/spec_results.txt', spec)
    np.savetxt('output/sens_results.txt', sens)
    np.savetxt('output/roc_results.txt', roc)

    print("Probabilities: norm: {0:.2f}, anem: {1:.2f}".format(prob_norm, prob_anem))
    print(np.around(spec.mean(), 3))
    print(np.around(spec.std(), 3))

    print(np.around(sens.mean(), 3))
    print(np.around(sens.std(), 3))

    print(np.around(roc.mean(), 3))
    print(np.around(roc.std(), 3))
    print("\n |-------------------------------------------| \n")


def clear_file(filename):
    # очистить файл, если он существует
    if os.path.exists(filename):
        f = open(filename, 'r+')
        f.truncate(0)
        f.close()
