import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from math import sqrt
from math import pi

def scatter_plot(X, Y, coef1, coef2, const):                                   # график рассеивания
    x = np.array([10, 35])                                                      # ось Х
    y_hyperplane = -(const + x * coef1) / coef2                                # построение дискриминирующей линии

    plt.figure()
    colors = ['darkorange', 'navy']
    target_names = ['normal', 'anemia']
    markers = ['o', 'X']

    plt.plot(x, y_hyperplane, 'k')

    for color, i, target_name, marker in zip(colors, [0, 1], target_names, markers):
        plt.scatter(X[Y == i, 0], X[Y == i, 1], alpha=.5, color=color, marker=marker, label=target_name)   # ОХ - недели, ОУ - КТИ
    plt.legend(loc='best', shadow=False)
    plt.title('The scattering of observations by features')
    plt.xlabel('Weeks')
    plt.ylabel('CTR')
    plt.show()

def plot_ROC(fpr_, tpr_, roc_auc_):
    """
    Ось Х обозначает 1 - специфичность (= fpr = FP/(FP+TN))
    Ось У обозначает чувствительность (= tpr = TP/(TP+FN))
    """

    plt.figure()
    lw = 2
    plt.plot(fpr_, tpr_, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_criteria_distributions(spec, sens, roc):
    """
    Создание весов для масштаба: 1 / длину списка.
    Все списки одинаковы по длине.
    Построение графиков по данным.
    """

    weights=np.ones_like(spec) / len(spec)

    fig, axs = plt.subplots(3, 1, sharey=True, sharex=True)
    axs[0].hist(roc, weights=weights)
    axs[0].set_xlabel('Специфичность')  

    axs[1].hist(sens, weights=weights)
    axs[1].set_xlabel('Чувствительность')

    axs[2].hist(roc, weights=weights)
    axs[2].set_xlabel('AUC')


def plot_cdf_edf(dist_name, x, cdf_fitted, edf_fitted, dir_name):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(dist_name)
    ax.plot(x, cdf_fitted,
            'r-', lw=3, alpha=0.6, label='Теоретическая')
    # отрисовка эмпирической
    ax.plot(x, edf_fitted,
            'b--', lw=3, alpha=0.6, label='Эмпирическая')
    ax.legend(loc='best', frameon=False)
    filename = dir_name + '/' + dist_name + '.png'
    plt.savefig(filename)
    #plt.show()

def plot_lda_vs_qda(lda, qda, X_test, y_test):
    """
    Отрисовка двух графиков рассеивания и областей классификации
    """
    target_names = ['anemia', 'normal']

    xx, yy = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    zz_lda = lda.predict_proba(X_grid)[:,1].reshape(xx.shape)
    zz_qda = qda.predict_proba(X_grid)[:,1].reshape(xx.shape)
    
    pl.figure()
    pl.subplot(1, 2, 1)
    pl.contourf(xx, yy, zz_lda > 0.5, alpha=0.5)
    pl.scatter(X_test[y_test==0,0], X_test[y_test==0,1], c='b', label=target_names[0])
    pl.scatter(X_test[y_test==1,0], X_test[y_test==1,1], c='r', label=target_names[1])
    pl.contour(xx, yy, zz_lda, [0.5], linewidths=2., colors='k')
    pl.legend()
    pl.axis('tight')
    pl.title('Linear Discriminant Analysis')
    
    pl.subplot(1, 2, 2)
    pl.contourf(xx, yy, zz_qda > 0.5, alpha=0.5)
    pl.scatter(X_test[y_test==0,0], X_test[y_test==0,1], c='b', label=target_names[0])
    pl.scatter(X_test[y_test==1,0], X_test[y_test==1,1], c='r', label=target_names[1])
    pl.contour(xx, yy, zz_qda, [0.5], linewidths=2., colors='k')
    pl.legend()
    pl.axis('tight')
    pl.title('Quadratic Discriminant Analysis')
    pl.show()

def plot_pdf_cdf_for_delong_test(Z, edf, cdf, pdf, params, plot_filepath, hist_filepath):
    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.plot(Z, cdf,
            'r-', lw=3, alpha=0.6, label='Теоретическая')
    ax.plot(Z, edf,
            'b--', lw=3, alpha=0.6, label='Эмпирическая')
    ax.set_title("Теоретическая и эмпирическая функции распределения", loc='center')
    ax.legend(loc='best')
    plt.savefig(plot_filepath)

    fig, ax = plt.subplots()
    ax.plot(Z, pdf)
    ax.hist(Z, 20, density=1)
    ax.set_title("Теоретическая и эмпирическая функции плотности распределения", loc='center')
    plt.savefig(hist_filepath)

def logist(x, loc, scale):
    return np.exp((loc-x)/scale)/(scale*(1+np.exp((loc-x)/scale))**2)

def plot_errors_dist():
    deviation = 0.0455
    fig = plt.figure(figsize =(5, 4)) 

    loc, scale = 0, (sqrt(3) * deviation / pi)
    s = np.random.logistic(loc, scale, 1200)
    #count, bins, ignored = plt.hist(s, bins=50, density=True)
    count, bins = np.histogram(s, bins=50, density=True)
    lgst_val = logist(bins, loc, scale)
    plt.plot(bins, lgst_val * count.max() / lgst_val.max(), color='k', label='logistic')

    # norm
    mu, sigma = 0, 0.0455 # mean and standard deviation
    x = np.linspace(-.2, .2, 100)
    #s = np.random.normal(mu, sigma, 4330)
    #count, bins, ignored = plt.hist(s, 100, density=True)
    count, bins = np.histogram(s, 50, density=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
              linewidth=2, color='r', label='norm')

    # laplace
    loc, scale = 0., (deviation / sqrt(2))
    #s = np.random.laplace(loc, scale, 16600)
    #count, bins, ignored = plt.hist(s, 50, density=True)
    x = np.linspace(-.2, .2, 100)
    pdf = np.exp(-abs(x-loc)/scale)/(2.*scale)
    plt.plot(x, pdf, label='laplace')


    deviation = 0.0455
    b = deviation * sqrt(3)
    a = -b
    s = np.random.uniform(a,b,4330)
    #count, bins, ignored = plt.hist(s, 50, density=True)
    count, bins,  = np.histogram(s, 50, density=False)
    
    plt.plot(bins, np.ones_like(bins) / (b - a), linewidth=2, color='m', label='uniform')
    
    Euler_Mascheroni_constant = 0.57721
    beta = sqrt(6) * deviation / pi
    mu = -Euler_Mascheroni_constant * beta
    #mu, beta = 0, 0.1 # location and scale
    s = np.random.gumbel(mu, beta, 16600)
   # count, bins, ignored = plt.hist(s, 50, density=True)
    count, bins  = np.histogram(s, 50, density=False)
    plt.plot(bins, (1/beta)*np.exp(-(bins - mu)/beta)
             * np.exp( -np.exp( -(bins - mu) /beta) ),
             linewidth=2, color='g', label='gumbel')

    plt.legend(loc="upper left")
    plt.show()


def plot_average_criteria_static():
    y_norm_spec = [1, 1, 1, 0.994, 0.983, 0.976, 0.969]
    y_norm_sens = [0.752, 0.777, 0.812, 0.839, 0.855, 0.876, 0.892]
    y_norm_roc = [0.963, 0.972, 0.982, 0.986, 0.988, 0.989, 0.988]
    
    y_rav_spec = [1, 1, 1, 1, 1, 0.995, 0.981]
    y_rav_sens = [0.695, 0.710, 0.743, 0.762, 0.785, 0.793, 0.813]
    y_rav_roc = [0.983, 0.989, 0.991, 0.992, 0.992, 0.992, 0.991]
    
    y_lapl_spec = [1, 1, 0.992, 0.985, 0.970, 0.965, 0.960]
    y_lapl_sens = [0.793, 0.814, 0.842,0.859, 0.885, 0.905, 0.921]
    y_lapl_roc = [0.963, 0.970, 0.979, 0.982, 0.983, 0.982, 0.981]

    y_log_spec = [1, 1, 0.994, 0.989, 0.974, 0.966, 0.961]
    y_log_sens = [0.777, 0.786, 0.813, 0.831, 0.857, 0.880, 0.896]
    y_log_roc = [0.964, 0.972, 0.982, 0.985, 0.986, 0.985, 0.983]
    
    y_gumb_spec = [1, 0.999, 0.991, 0.984, 0.968, 0.957, 0.952]
    y_gumb_sens = [0.746, 0.784, 0.838, 0.867, 0.895, 0.914, 0.927]
    y_gumb_roc = [0.997, 0.997, 0.994, 0.992, 0.987, 0.984, 0.981]


    x = [0.1, 0.2, 0.4, 0.535, 0.8, 1, 1.2]

    plt.plot(x, y_norm_spec, label='norm', color='b')
    plt.plot(x, y_rav_spec, label='uniform', color='r')
    plt.plot(x, y_lapl_spec, label='laplace', color='g')
    plt.plot(x, y_log_spec, label='logistic', color='y')
    plt.plot(x, y_gumb_spec, label='gumbel', color='k')
    plt.legend(loc='best')
    plt.title('Специфичность')
    plt.xlabel('σ(н.) / σ(а.)')
    plt.show()

    plt.plot(x, y_norm_sens, label='norm', color='b')
    plt.plot(x, y_rav_sens, label='uniform', color='r')
    plt.plot(x, y_lapl_sens, label='laplace', color='g')
    plt.plot(x, y_log_sens, label='logistic', color='y')
    plt.plot(x, y_gumb_sens, label='gumbel', color='k')
    plt.legend(loc='best')
    plt.title('Чувствительность')
    plt.xlabel('σ(н.) / σ(а.)')
    plt.show()
    
    plt.plot(x, y_norm_roc, label='norm', color='b')
    plt.plot(x, y_rav_roc, label='uniform', color='r')
    plt.plot(x, y_lapl_roc, label='laplace', color='g')
    plt.plot(x, y_log_roc, label='logistic', color='y')
    plt.plot(x, y_gumb_roc, label='gumbel', color='k')
    plt.legend(loc='best')
    plt.title('ROC AUC')
    plt.xlabel('σ(н.) / σ(а.)')
    plt.show()
