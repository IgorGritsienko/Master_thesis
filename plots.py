import matplotlib.pyplot as plt
import numpy as np


def scatter_plot(X, y, coef1, coef2, const):
    # ось Х по неделям, берется минимум и максимум из существующих -+1
    x = np.array([X[:, 0].min() - 1, X[:, 0].max() + 1])
    # построение дискриминирующей линии
    y_hyperplane = -(const + x * coef1) / coef2

    plt.figure()
    colors = ['darkorange', 'navy']
    target_names = ['normal', 'anemia']
    markers = ['o', 'X']

    plt.plot(x, y_hyperplane, 'k')

    for color, i, target_name, marker in zip(colors, [0, 1], target_names, markers):
        plt.scatter(X[y == i, 0], X[y == i, 1], alpha=.5, color=color, marker=marker,
                    label=target_name)  # ОХ - недели, ОУ - КТИ
    plt.legend(loc='best', shadow=False)
    plt.title('The scattering of observations by features')
    plt.xlabel('Weeks')
    plt.ylabel('CTR')
    plt.show()


def plot_roc(fpr_, tpr_, roc_auc_):
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

    weights = np.ones_like(spec) / len(spec)

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


def plot_pdf_cdf_for_delong_test(z, edf, cdf, pdf, plot_filepath, hist_filepath):
    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.plot(z, cdf,
            'r-', lw=3, alpha=0.6, label='Теоретическая')
    ax.plot(z, edf,
            'b--', lw=3, alpha=0.6, label='Эмпирическая')
    ax.set_title("Теоретическая и эмпирическая функции распределения", loc='center')
    ax.legend(loc='best')
    plt.savefig(plot_filepath)

    fig, ax = plt.subplots()
    ax.plot(z, pdf)
    ax.hist(z, 20, density=1)
    ax.set_title("Теоретическая и эмпирическая функции плотности распределения", loc='center')
    plt.savefig(hist_filepath)
