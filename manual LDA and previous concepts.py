import numpy as np

def CovarianceMatrix(X):
    Xt = np.transpose(X)                                            # транспонирование
    S = np.dot(Xt, X)                                               # Xt * X
    #S = np.divide(S, np.ma.size(X, 0))                             # умножить на 1/n , n - количество записей для группы
    return S

def JointCovarianceMatrix(S1, S2, n1, n2):
    #S = np.add(np.multiply(S1, np.ma.size(X1, 0)) , np.multiply(S2, np.ma.size(X2, 0)))                                              # S1 + S2
    S = np.add(S1, S2)
    S = np.divide(S, (n1 + n2))                                     # делим матрицу на (количество записей в обеих группах минус 2)
    return S

def DiscriminantFunctionCoefficientVector(reversedS, deltaX):
   A = np.dot(reversedS, deltaX)                                    # умножаем S^(-1) на разницу средних значений
   return A

def DiscriminantFunctionValueVector(X, A):                          # получаем вектор значений дискриминантной функции для одной группы
    f = np.dot(X, A)                                                # умножаем матрицу наблюдений на вектор коэффициентов дискриминантной функции
    return f
     
def DiscriminantionConstant(f1_avg, f2_avg):                        # получаем константу дискриминации
    c = (f1_avg + f2_avg) / 2.0
    return c

def CovarianceMatrixElems_StandardCoeff(X1, X2):
    W = np.array([])                                                # массив для хранений значений элементов матрицы W (хранятся Wjj) 
    for i in range(np.ma.size(X1, 1)):                                          # количество столбцов
        Wjj = np.inner(X1[:, i], X1[:, i]) + np.inner(X2[:, i], X2[:, i])       # Wjj = {sum(k = 1, m)    [sum(i = 1, Nk)   (Xikj - averageXkj)(Xikj - averageXkj)]}
                                                                                                # в коде в цикле по каждому столбцу берем все строки наблюдений матрицы Х (средние значения уже вычли) умножаем на этот же элемент.
                                                                                                # к этому прибавляем тоже самое, но со второй группы. 
                                                                                                # количество столбцов = кол-во переменных = кол-во коэффициентов
                                                                                              
        W = np.append(W, Wjj)
    return W

def StandardCoefficients(A, W, p):                                  # подсчет стандартных коэффициентов
    B = np.array([])
    m = 2                                                           # число групп
    for i in range(np.ma.size(A, 0)):
        B = np.append(B, A[i] * np.sqrt(W[i] / (p - m)))            # bj = aj * sqrt(Wj / (p-m))
    return B

def LDA(normalCTR, anemiaCTR):

    # ИСПОЛЬЗОВАЛОСЬ ДЛЯ ПРОВЕРКИ КОРРЕКТНОСТИ РАБОТЫ ПРОГРАММЫ - СОВПАЛО С КНИГОЙ 
    #normalCTR = np.array([[0.15, 0.34, 0.09, 0.21], [1.91, 1.68, 1.89, 2.30]]).T
    #anemiaCTR = np.array([[0.48, 0.41, 0.62, 0.50, 1.20], [0.88, 0.62, 1.09, 1.32, 0.68]]).T

    normalAverageVariables = np.mean(normalCTR, 0)                      # среднее значение для каждого (из двух) параметра на основе наблюдений для первой группы
    anemiaAverageVariables = np.mean(anemiaCTR, 0)                      # среднее значение для каждого (из двух) параметра на основе наблюдений для второй группы


    # Вычли из каждого элемента соответствующие средние
    X1 = np.subtract(normalCTR, normalAverageVariables)                 # вычитаем и знаблюдений соответствующие средние значение по каждому параметру
    X2 = np.subtract(anemiaCTR, anemiaAverageVariables)     

    S1 = CovarianceMatrix(X1)                                           # ищем ковариационную матрицу Xt*X*1/n
    S2 = CovarianceMatrix(X2)

    n1 = np.ma.size(X1, 0)                                              # количество наблюдений в матрице Х
    n2 = np.ma.size(X2, 0)

    S = JointCovarianceMatrix(S1, S2, n1, n2)                           # совместная ковариационная матрица (S1+S2) / (n1 + n2 - 2), n1, n2 - количество наблюдений в 1 и 2 группах

    reversedS = np.linalg.inv(S)                                        # S^(-1)

    deltaAverageVariables = np.subtract(normalAverageVariables, anemiaAverageVariables)     # находим разницу между средними значениями наблюдений разных групп

    A = DiscriminantFunctionCoefficientVector(reversedS, deltaAverageVariables)             # вектор коэффициентов дискриминатной функции

    f1 = DiscriminantFunctionValueVector(normalCTR, A)                                      # вектор значений дискриминантной функции для первого и второго подмножеств
    f2 = DiscriminantFunctionValueVector(anemiaCTR, A)

    f1_average = np.mean(f1, 0)                                                             # средние значения дискриминантной функции
    f2_average = np.mean(f2, 0)

    c = DiscriminantionConstant(f1_average, f2_average)                                     # константа дискриминации
    W = CovarianceMatrixElems_StandardCoeff(X1, X2)                                         # находим элементы Wjj
    p = 2                                                                                   # общее количество исходных переменных
    #B = StandardCoefficients(A, W, p)                                                      # стандартизированные коэффициенты
    print('Коэффициент при параметре недель: {: .30f}\nКоэффициент при параметре КТИ: {: .30f}\nКонстанта дискриминации c: {: .87f}'.format(A[0], A[1], c))
    #print('Стандартизированный оэффициент при параметре недель: {: .30f}\nСтандартизированный коэффициент при параметре КТИ: {: .30f}\nКонстанта дискриминации c: {: .87f}'.format(B[0], B[1], c))
    return A[0], A[1], c

def FileWrite(filename, data):                                                                          # запись в файл .txt
    np.savetxt(filename, data, fmt='%.18g', header = "Стандартизированные коэффициенты, полученные в процессе дискриминантного анализа. Переменная при коэффициенте указана в названии файла.")

    # было в цикле с запусками всей процедуры
    #coefCTR, coefWeek, constant = LDA(normal_CTR, anemia_CTR)                                         # получили значение коэффициента КТИ, недели и переменную дискриминации

    #coefDistribution = np.append(coefDistribution, [coefCTR, coefWeek, constant])

#coefDistribution = np.reshape(coefDistribution, (simulationAmount, 3))
    
#CTRDistr = np.insert(coefDistribution[:, 0], 0, [0, simulationAmount])
#weekDistr =  np.insert(coefDistribution[:, 1], 0, [0, simulationAmount])
#constantDistr = np.insert(coefDistribution[:, 2], 0, [0, simulationAmount])
#FileWrite("CTRcoef.txt",CTRDistr)                                                    # коэффициенты при кти
#FileWrite("weekCoef.txt", weekDistr)                                                   # коэффициенты при неделях
#FileWrite("constants.txt", constantDistr)                                                  # константа


#fig0 = plt.figure()
##fig, axes = plt.subplots(1, 3)             если  необходимо в одной фигуре нарисовать все графики (не друг на друге, а только в пределах одного окна, графики будут рядом друг с другом находиться)
##ax0, ax1, ax2 = axes.flatten()
#ax0 = fig0.add_subplot(111)
#ax0.hist(coefDistribution[:, 0], bins = 10, density=True, histtype='bar', color = 'violet')
#ax0.set_title('Распределение')
#ax0.set_xlabel('Values')
#ax0.set_ylabel('Probability')
#
#fig1 = plt.figure()
#ax1 = fig1.add_subplot(111)
#ax1.hist(coefDistribution[:, 1], bins = 10, density=True, histtype='bar', color = 'crimson')
#ax1.set_title('Распределение')
#ax1.set_xlabel('Values')
#ax1.set_ylabel('Probability')
#
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#ax2.hist(coefDistribution[:, 2], bins = 10, density=True, histtype='bar', color = 'palegreen')
#ax2.set_title('Распределение')
#ax2.set_xlabel('Values')
#ax2.set_ylabel('Probability')
#
#fig0.tight_layout()
#fig1.tight_layout()
#fig2.tight_layout()
#
#plt.show()
#
#
#
#
##fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
#
##axs[0].hist(coefDistribution[:, 0], bins = 10, density = True, color = 'violet')
##axs[1].hist(coefDistribution[:, 1], bins = 10, density = True, color = 'crimson')
##axs[2].hist(coefDistribution[:, 2], bins = 10, density = True, color = 'palegreen')







    # необходимость в использовании отпала, но фильтрация данных полезная


    #onlyNormal_Xtest = np.array(Xy[N_train_rate:, :][Xy[N_train_rate:, :][:, 2] == 0])         # arr = arr[arr == 0]       только нормальные Х
    #onlyAnemia_Xtest =  np.array(Xy[N_train_rate:, :][Xy[N_train_rate:, :][:, 2] == 1])        # только анемичные Y
    #onlyNormal_Xtest = np.delete(onlyNormal_Xtest, 2, 1)                             # удаляем столбец с Y
    #onlyAnemia_Xtest = np.delete(onlyAnemia_Xtest, 2, 1)