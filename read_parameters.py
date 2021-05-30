def Read_parameters(filename):
    parameters = []
    with open(filename, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parameters.append(line)   
    parameters = [x.rstrip("\n") for x in parameters]                          # удалить '/n' символы из элементов массива
    return parameters
