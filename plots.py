import numpy as np
import matplotlib.pyplot as plt

def scatter_plot(X, Y, coef1, coef2, const):                                   # график рассеивания
    x = np.array([-2, 2])                                                      # ось Х
    y_hyperplane = -(const + x * coef1) / coef2                                # построение дискриминирующей линии

    plt.figure()
    colors = ['navy', 'darkorange']
    target_names = ['anemia', 'normal']

    plt.plot(x, y_hyperplane, 'k')

    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(X[Y == i, 0], X[Y == i, 1], alpha=.5, color=color, label=target_name)   # ОХ - недели, ОУ - КТИ
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
    plt.show()

