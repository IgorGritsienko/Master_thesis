import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
n = 100


#print(a)
x = np.array([0.88888889, 0.90163934, 0.9047619 , 0.92727273, 0.93103448,
       0.93103448, 0.93103448, 0.93220339, 0.95081967, 0.96610169,
       0.97058824, 0.98076923, 0.98076923, 0.98148148, 0.98181818])



mu = 0
df = 2.74
#x = np.linspace(0, 7, 50)
x = np.linspace(st.t.ppf(0.01, df), st.t.ppf(0.99, df), 100)
cdf = st.t.cdf(x, df)
plt.plot(x, cdf)
print(cdf)
#data = f(x,0.2,1) + 0.05*np.random.randn(n)



    # Строим гистограмму, по ней определяем минимальные и максимальные значения по ОХ, 
    # чтобы далее на этом участке построить теоретическую плотность
    #fig = plt.figure()
    #ax = fig.add_axes([.775, .125, .775, .755])
    #xt = ax.get_xticks()
    #xt = plt.xticks()[0]
    #xmin, xmax = min(xt), max(xt)
    #lnspc = np.linspace(xmin, xmax, len(test_data))

   # cdf_fitted = dist.cdf(lnspc, *params)
    #ax.plot(lnspc, cdf_fitted, label=dist_name)
    #ax.plot(edf, lnspc)
    #ax.hist(edf, lnspc - 1, density=True, histtype='step', cumulative=True, label = 'Эмпирическая')