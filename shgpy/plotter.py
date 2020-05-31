import matplotlib.pyplot as plt
import numpy as np


def easy_plot(list_of_dats, list_of_param_dicts, pcs_to_include, show_plot=True, filename=None, show_legend=False):
    n = len(pcs_to_include)
    nrows = [1, 1, 1, 2][n-1]
    ncols = [1, 2, 3, 2][n-1]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, squeeze=False)
    for pc, ax in zip(pcs_to_include, axs.flatten()):
        for dat, kwparams in zip(list_of_dats, list_of_param_dicts):
            xdata, ydata = dat.get_pc(pc, 'radians')
            ax.plot(xdata, ydata, **kwparams)
        ax.set_title(pc)
        if show_legend:
            ax.legend(loc='upper right')
    if filename is not None:
        plt.savefig(filename)
    if show_plot:
        plt.show()


def easy_polar_plot(list_of_dats, list_of_param_dicts, pcs_to_include, show_plot=True, filename=None, show_legend=False):

    n = len(pcs_to_include)
    fig, axs = plt.subplots(1, n, subplot_kw=dict(projection='polar'), sharey=True)

    for pc, ax in zip(pcs_to_include, axs.flatten()):
        for dat, kwparams in zip(list_of_dats, list_of_param_dicts):
            xdata, ydata = dat.get_pc(pc, 'radians')
            ax.plot(xdata, ydata, **kwparams)
        ax.set_title(pc)
        if show_legend:
            ax.legend(loc='upper right')
    for pc,ax in zip(pcs_to_include, axs):
        ax.spines['polar'].set_visible(False)
        ax.set_yticklabels([])
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        ax.set_xticklabels(['', '', '', '', ''])
        ax.set_rticks([max([dat.get_maxval()[1] for dat in list_of_dats])])
        ax.set_rmin(0)
        ax.set_title(pc)
        ax.set_facecolor('#ffffff')
    if filename is not None:
        plt.savefig(filename)
    if show_plot:
        plt.show()

    













