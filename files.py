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


def Read_csv(filename, n_rows, position):
    # sep=',' для записи в один столбец через ','
    df = pd.read_csv(filename, nrows=n_rows, skiprows=position, sep=';', header=0)
    return df

def Save_sample_to_csv(array, filename, _mode):
    df = pd.DataFrame(data={"Weeks": array[:, 0], "CTR": array[:, 1], "target": array[:, 2]})       # 3 столбца датафрейма: КТИ, недели и класс (анемия или норма)
    df.to_csv(filename, sep=';',index=False, mode=_mode, header=False, float_format='%.6f')              # mode 'a' - добавление новых записей к старым, создание файла при его отсутствии. 'w' - перезапись



def save_tests_results_to_csv(compbdt_data, filename):
    df = pd.DataFrame(data={"s11": compbdt_data[:, 0], "s10": compbdt_data[:, 1], 
                            "s01": compbdt_data[:, 2], "s00": compbdt_data[:, 3],
                            "r11": compbdt_data[:, 4], "r10": compbdt_data[:, 5],
                            "r01": compbdt_data[:, 6], "r00": compbdt_data[:, 7]})
    df.to_csv(filename, sep=',', index = False, mode='a', header=not os.path.exists(filename))
    


def quality_criteria_results_print_save_to_file(spec, sens, roc, prob_norm, prob_anem, sigma_ratio, simAmount):
    """
    Сохраняем выборки специфичности, чувствительности и рок аук в файл.
    Название файла соответствует отношение сигм и числу экспериментов.
    """

    np.savetxt("output/spec_res_ratio_" + str(sigma_ratio) + "_amount_" + str(simAmount) + ".txt", spec)
    np.savetxt("output/sens_res_ratio_" + str(sigma_ratio) + "_amount_" + str(simAmount) + ".txt", sens)
    np.savetxt("output/roc_res_ratio_" + str(sigma_ratio) + "_amount_" + str(simAmount) + ".txt", roc)

def clear_file(filename):
    """
    Очистить файл, если он существует
    """
    
    if os.path.exists(filename):
        f = open(filename, 'r+')
        f.truncate(0)
        f.close()
        
def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)

def create_dir(dir_name):
    """
    Создать папку, если таковой еще не существует
    """
    
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        
def write_to_txt(filename, array):
        with open(filename, 'w') as f:
            for item in array:
                f.write("%s\n" % item)