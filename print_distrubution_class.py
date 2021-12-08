import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st

def main_process(dist_name):
    filename = dist_name + '.csv'
    df = pd.read_csv(filename, delimiter=';')
    sigma_ratio = df.iloc[:, 0].values
    
    params = []
    for i in df.iloc[:, 1].values:
        params.append(tuple(map(float, i[1 : -1 : ].split(', '))))

    fig, ax = plt.subplots(figsize=(8, 4))
    #ax.set_title(dist_name)
    x = np.linspace(0.88, 1.0, 100)

    for i in range(df.shape[0]):
        dist = getattr(st, dist_name)
        
        cdf_fitted = dist.cdf(x, *params[i][:-2], loc=params[i][-2], scale=params[i][-1])
        ax.plot(x, cdf_fitted, lw=3, alpha=0.6, label=sigma_ratio[i])
        # отрисовка эмпирической
    ax.legend(loc='best', frameon=False)
    plt.show()


if __name__ == "__main__":
    # Начало программы
    #  для фиксации значений выставляем seed
    np.random.seed(0)
    warnings.filterwarnings("ignore")
    sns.set_style("darkgrid")
    
    # x = np.linspace(0, 1.0, 100)
    # fig, ax = plt.subplots(figsize=(8, 4))
    # dist = getattr(st, "gumbel_l")
    # cdf_fitted = dist.cdf(x,loc=0.98, scale=0.01)
    # ax.plot(x, cdf_fitted, lw=3, alpha=0.6)
    # plt.show()
    
    dist_name = "exponpow"
    #"pearson3"
    #"exponpow"
    #"johnsonsb"
    main_process(dist_name)