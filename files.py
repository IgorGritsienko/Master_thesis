import os
import shutil

import numpy as np
import pandas as pd


def read_parameters(filename):
    """Чтение строк из файла за исключением комментариев."""
    parameters = []
    with open(filename, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parameters.append(line)
    parameters = [x.rstrip("\n") for x in parameters]  # удалить '/n' символы из элементов массива
    return parameters


def read_csv(filename, rows_number, position):
    # sep=',' для записи в один столбец через ','
    df = pd.read_csv(filename, nrows=rows_number, skiprows=position, sep=';', header=0)
    return df


def save_sample_to_csv(array, filename, save_mode):
    # 3 столбца датафрейма: КТИ, недели и класс (анемия или норма)
    df = pd.DataFrame({"Weeks": array[:, 0],
                       "CTR": array[:, 1],
                       "target": array[:, 2]})
    df.to_csv(filename,
              sep=';',
              index=False,
              mode=save_mode,
              header=False,
              float_format='%.6f')


def clear_file(file_name):
    """Очистить файл, если он существует."""
    if os.path.exists(file_name):
        f = open(file_name, 'r+')
        f.truncate(0)
        f.close()


def create_dir(dir_name):
    """Создать папку, если таковой еще не существует."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def delete_file(file_name):
    """Удалить файл, если он существует."""
    if os.path.exists(file_name):
        os.remove(file_name)


def delete_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)


def write_to_txt(file_name, array):
    """Сохранить массив в файл."""
    with open(file_name, 'w') as f:
        for item in array:
            f.write("%s\n" % item)


def write_quality_criteria(path, distr, spec, sens, roc, sigma_ratio, simulation_amount):
    """
    Сохраняем выборки специфичности, чувствительности и рок аук в файл.
    Название файла соответствует отношению среднеквадратичных отклонений и числу экспериментов.
    """
    np.savetxt(path + "/" + distr + "/spec_" + str(sigma_ratio) + "_" + str(simulation_amount) + ".txt", spec)
    np.savetxt(path + "/" + distr + "/sens_" + str(sigma_ratio) + "_" + str(simulation_amount) + ".txt", sens)
    np.savetxt(path + "/" + distr + "/roc_" + str(sigma_ratio) + "_" + str(simulation_amount) + ".txt", roc)


def write_isw(file_name, array, comment):
    """Сохранить массив в файл по требуемому ISW формату."""
    with open(file_name, 'w') as f:
        f.write("%s\n" % comment)
        f.write("%s\n" % str(0))
        f.write("%s\n" % str(len(array)))
        for item in array:
            f.write("%s\n" % item)
