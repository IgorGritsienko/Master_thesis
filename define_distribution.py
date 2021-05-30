import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

def get_best_distribution(data, number_distributions_to_plot):
    
    # Set up empty lists to stroe results
    D_values = []
    p_values = []
    parameters = []

    weights=np.ones_like(data) / len(data)
    # Строим гистограмму, по ней определяем минимальные и максимальные значения по ОХ, чтобы далее на этом участке построить теоретическую плотность
    plt.hist(data, weights=weights)
    xt = plt.xticks()[0]
    xmin, xmax = min(xt), max(xt)
    lnspc = np.linspace(xmin, xmax, len(data))
# =============================================================================
#     dist_names = ["alpha","anglit","arcsine","beta","betaprime","bradford","burr","cauchy","chi","chi2","cosine",
#         "dgamma","dweibull","erlang","expon","exponnorm","exponweib","exponpow","f","fatiguelife","fisk",
#         "foldcauchy","foldnorm","genlogistic","genpareto","gennorm","genexpon",
#         "genextreme","gausshyper","gamma","gengamma","genhalflogistic","gilbrat","gompertz","gumbel_r",
#         "gumbel_l","halfcauchy","halflogistic","halfnorm","halfgennorm","hypsecant","invgamma","invgauss",
#         "invweibull","johnsonsb","johnsonsu","ksone","kstwobign","laplace","levy","levy_l","levy_stable",
#         "logistic","loggamma","loglaplace","lognorm","lomax","maxwell","mielke","nakagami","ncx2","ncf",
#         "nct","norm","pareto","pearson3","powerlaw","powerlognorm","powernorm","rdist","reciprocal",
#         "rayleigh","rice","recipinvgauss","semicircular","t","triang","truncexpon","truncnorm","tukeylambda",
#         "uniform","vonmises","vonmises_line","wald","weibull_min","weibull_max","wrapcauchy"]
# =============================================================================
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]

        #_distn_names
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)
        
        # Obtain the KS test P statistic, round it to 5 decimal places
        D, p = st.kstest(data, dist_name, args=param)
        p = np.around(p, 5)
        p_values.append(p)
        D_values.append(D)

    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['D_stats'] = D_values
    results['p_value'] = p_values
    results.sort_values(['D_stats'], inplace=True)

    print(results)

    dist_names = results['Distribution'].iloc[0:number_distributions_to_plot]

    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)
        parameters.append(param)
                           
        pdf_fitted = dist.pdf(lnspc, *param[:-2], loc=param[-2], scale=param[-1]) * weights
        
        #scale_pdf = np.trapz (h[0], h[1][:-1]) / np.trapz (pdf_fitted, lnspc)
        #pdf_fitted *= scale_pdf
        plt.plot(lnspc, pdf_fitted, label=dist_name)
    #plt.plot(lnspc, dist.pdf(lnspc, param[-2], param[-1]))
    dist_parameters = pd.DataFrame()
    
    dist_parameters['Distribution'] = (results['Distribution'].iloc[0:number_distributions_to_plot])
    dist_parameters['Distribution parameters'] = parameters

    plt.legend(loc='upper right')
    plt.show()

    # Print parameter results
    print ('\nDistribution parameters:')
    print ('------------------------')
    
    for index, row in dist_parameters.iterrows():
        print ('\nDistribution:', row[0])
        print ('Parameters:', row[1])
    return dist_parameters.iat[0, 0], dist_parameters.iat[0, 1]
