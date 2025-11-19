import numpy as np
import pandas as pd
from scipy import interpolate
from matplotlib import pyplot as plt
from PIL import Image
from utils import unit
file = '../result/'
# mol = 'ttm/'
# mol = 'bispytm/'
# mol = 'ttm3ncz/'
# mol = 'ptm3ncz/'
# mol = 'mttm2/'
# mol = 'hhcrqpp2/'
mol = 'g3ttm/'
# mol = 'c6h5nit/'
if mol == 'hhcrqpp2/':
    rows = 300
elif mol == 'g3ttm/':
    rows = 500
else:
    rows = 100
# colors = ["#0B7055", "#796CAD", "#D65813", "#23BAC5", "#EECA40", "#FFA070",
#           "#000080", "#000000", "#6888F5", "#D77071", "#F0C284", "#EF8B67",
#           "#808080", "#85C3DC"]
# colors = ["#547bb4", "#629c35", "#6c61af", "#6f6f6f", "#c0321a", "#dd7c4f", "#000000"]
# colors = ["#EF2C2B", "#EEBEC0", "#23B2E0", "#A5CC5B", "#C2C2C2", "#000000"]
colors_doc = ["#EF2C2B", "#EF2C2B", "#23B2E0", "#23B2E0", "#A5CC5B", "#000000"]  # doctor paper
colors = ["#EF2C2B", "#EF2C2B", "#23B2E0", "#23B2E0", "#000000"]  # paper
colors_functionals = ["#B31E1D", "#EF2C2B", "#F56664", "#1881A7", "#23B2E0", "#6FD0F0", "#000000"]
solvent = '-TOLUENE'
# solvent = '-CYCLOHEXANE'
# solvent = '-ACETONITRILE'
# solvent = '-METHANOL'
functional = 'pbe0-st-10eV/'
# functional = 'b3lyp/'
# functional = 'pbe0-st/'
# functional = 'tpssh-st-10eV/'

# plt Settings
plt.rc('font', family='Times New Roman', size=14)
plt.rcParams['mathtext.fontset'] = 'custom'

plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'
plt.rcParams['mathtext.rm'] = 'STIXGeneral:bold'
plt.rcParams['mathtext.tt'] = 'STIXGeneral'
plt.rcParams['mathtext.bf'] = 'STIXGeneral:bold:italic'
plt.rcParams['mathtext.cal'] = 'STIXGeneral'
plt.rcParams['mathtext.sf'] = 'STIXGeneral'

plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.linewidth'] = 1


def UVvis(*args, ds2, num_method, labels, colors, fwhm=0.3, title='UVspec', minx=300, maxx=700, maxy2=1, norm=False):
    """this function divide X and U to two figure"""
    in_num = np.sum(num_method)
    assert in_num == len(labels), "input data and label must have same length"
    # for l in labels:
    #     if 'Exp.' in l:
    #         in_num -= 1
    minx = np.inf
    maxx = -np.inf
    for i in range(in_num):
        if min(args[2*i]) < minx:
            minx = min(args[2*i])
        if max(args[2*i]) > maxx:
            maxx = max(args[2*i])
    x = np.linspace(minx - 25, maxx + 25, 1000)  # nm as unit
    y = np.zeros((x.shape[0], in_num))

    def gaussian(x, x0, A, fwhm):
        wavenum = 1 / (unit.eVxnm / fwhm)
        return 1.3062974e8 * A / (1e7 * wavenum) * np.exp(-((1 / x - 1 / x0) / (1 * wavenum)) ** 2)

    for i in range(in_num-1):  # minor 1 is because last energy and oscillator is experiment data
        for pos, height in zip(args[2 * i], args[2 * i + 1]):
            y[:, i] += gaussian(x, pos, height, fwhm)
        if norm:
            y[:, i] = y[:, i] / np.max(y[:, i])

    fig, (ax1, ax2, ax3) = plt.subplots(len(num_method), 1, figsize=(8, 12), sharex=True)
    # ax2 = ax.twinx()
    # experiment
    for i in range(num_method[2]):
        ax1.plot(args[-2], args[-1], color=colors[-1], lw=2, label=labels[-1])
        # nlabel += 1
    # calculation
    for i, l, c in zip(range(num_method[0]), labels[:num_method[0]], colors[:num_method[0]]):
        ax1.plot(x, y[:, i], color=c, lw=2, label=l)
        # nlabel += 1
    # experiment
    for i in range(num_method[2]):
        ax2.plot(args[-2], args[-1], color=colors[-1], lw=2, label=labels[-1])
    # calculation
    for i, l, c in zip(range(num_method[1]), labels[num_method[0]:np.sum(num_method[:2])], colors[num_method[0]:np.sum(num_method[:2])]):
        ax2.plot(x, y[:, num_method[0]+i], color=c, lw=2, label=l)
    # # peak position
    # for i in range(in_num):
    #     for j, (pos, height) in enumerate(zip(args[2*i], args[2*i+1])):
    #         ax2.vlines(pos, 0, height, color=colors[i], linestyle="dashed", alpha=0.6,
    #                    label=labels[i] if j == 0 else None)
    #     # nlabel += 1
    # # for i, (pos, height) in enumerate(zip(args[0], args[1])):
    # #     ax2.vlines(pos, 0, height, color=colors[0], linestyle="dashed", alpha=0.6,
    # #                label=label[0]+' $f$' if i == 0 else None)  # only for first method
    # ax2.set_ylabel("$f$", fontsize=15)
    # ax2.tick_params(axis="both", which="major", labelsize=15)
    # ax2.set_ylim([0, maxy2])
    ax3.bar(ds2[0], ds2[1], width=1.5, label=ds2[3], color=ds2[2])

    ax3.set_xlim(min(x), max(x))
    ax3.set_xlabel("wavelength (nm)", fontsize=15)
    # ax.set_ylabel(r"$\epsilon$ / $M^{-1}cm^{-1}$", fontsize=15)
    if norm:
        ax1.set_ylabel(r"norm. $\epsilon$", fontsize=15)
        ax2.set_ylabel(r"norm. $\epsilon$", fontsize=15)
    else:
        ax1.set_ylabel(r"$\epsilon$ / $M^{-1}cm^{-1}$", fontsize=15)
        ax2.set_ylabel(r"$\epsilon$ / $M^{-1}cm^{-1}$", fontsize=15)
    ax1.tick_params(axis="both", which="major", labelsize=15)
    ax3.set_ylabel(r'$\Delta S^2$', fontsize=15)
    maxy = np.max((np.max(args[-1]), np.max(y)))
    # # print(maxy)
    # if not norm:
    #     if mol == 'ttm/':
    #         maxy = 41476.99925456449  # ttm fwhm=0.2
    #     elif mol == 'bispytm/':
    #         maxy = 29125.924840564476  # bispytm fwhm=0.2
    #     elif mol == 'ttm3ncz/':
    #         maxy = 62891.109187797156  # ttm3ncz fwhm=0.2
    #     elif mol == 'ptm3ncz/':
    #         maxy = 42615.05030815971  # ptm3ncz fwhm=0.2
    #     elif mol == 'mttm2/':
    #         maxy = 83331.98247503756  # mttm2 fwhm=0.2
    #     elif mol == 'hhcrqpp2/':
    #         maxy = 70018.44883471262  # hhcrqpp2 fwhm=0.2  row=300
    ax1.set_ylim([0, 1.1 * maxy])
    ax2.set_ylim([0, 1.1 * maxy])
    ax3.set_ylim(0, 2.1)
    # ax1.grid(True, linestyle="--", alpha=0.5)
    # ax2.grid(True, linestyle="--", alpha=0.5)
    # ax3.grid(True, linestyle="--", alpha=0.5)
    # # combine legend
    # lines1, labels1 = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax.legend(lines1 + lines2, labels1 + labels2, fontsize=15, loc='upper right')
    # ax.legend(fontsize=15)
    ax1.legend(framealpha=1)
    ax2.legend(framealpha=1)
    ax3.legend(framealpha=1)
    fig.tight_layout()

    # all_lines = lines1 + lines2
    # all_labels = labels1 + labels2
    # ax.legend(all_lines, all_labels,
    #       loc='lower center',
    #       bbox_to_anchor=(0.5, 1.02),
    #       ncol=np.ceil(nlabel/2),
    #       columnspacing=1.5,
    #       handlelength=2,
    #       fontsize=15)
    plt.axes([0.6, 0.65, 0.3, 0.3])  # ttm
    bgimg = plt.imread(file + mol + functional + mol[:-1] + '.bmp')
    plt.imshow(bgimg)
    plt.axis("off")
    plt.axes([0.6, 0.33, 0.3, 0.3])  # ttm
    bgimg = plt.imread(file + mol + functional + mol[:-1] + '.bmp')
    plt.imshow(bgimg)
    plt.axis("off")
    plt.show()
    fig.savefig(title + '.eps', dpi=1200, bbox_inches='tight')
    fig.savefig(title + '.pdf', dpi=1200, bbox_inches='tight')


def UVvis2(*args, ds2, num_method, labels, colors, fwhm=0.3, title='UVspec', maxy2=1, norm=False):
    '''this function combine U and X in single figure'''
    in_num = np.sum(num_method)
    assert in_num == len(labels)-1, "input data and label must have same length"
    # for l in labels:
    #     if 'Exp.' in l:
    #         in_num -= 1
    minx = np.inf
    maxx = -np.inf
    for i in range(in_num+1):
        if min(args[2*i]) < minx:
            minx = min(args[2*i])
        if max(args[2*i]) > maxx:
            maxx = max(args[2*i])
    x = np.linspace(minx - 25, maxx + 25, 1000)  # nm as unit
    y = np.zeros((x.shape[0], in_num))

    def gaussian(x, x0, A, fwhm):
        wavenum = 1 / (unit.eVxnm / fwhm)
        return 1.3062974e8 * A / (1e7 * wavenum) * np.exp(-((1 / x - 1 / x0) / (1 * wavenum)) ** 2)

    for i in range(in_num):  # minor 1 is because last energy and oscillator is experiment data
        for pos, height in zip(args[2 * i], args[2 * i + 1]):
            y[:, i] += gaussian(x, pos, height, fwhm)
        if norm:
            y[:, i] = y[:, i] / np.max(y[:, i])

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    ax2 = ax1.twinx()
    # experiment
    ax1.plot(args[-2], args[-1], color=colors[-1], lw=2, label=labels[-1])
    # nlabel += 1
    # calculation
    for i, l, c in zip(range(in_num), labels[:in_num], colors[:in_num]):
        if i in [1,3,4]:
            ax1.plot(x, y[:, i], '--', color=c, lw=2, label=l)
        if i in [0,2]:
            ax1.plot(x, y[:, i], color=c, lw=2, label=l)
        # nlabel += 1
    # # peak position
    # for i in range(in_num):
    #     for j, (pos, height) in enumerate(zip(args[2*i], args[2*i+1])):
    #         ax2.vlines(pos, 0, height, color=colors[i], linestyle="dashed", alpha=0.6,
    #                    label=labels[i] if j == 0 else None)
    #     # nlabel += 1
    for i, (pos, height) in enumerate(zip(args[2], args[3])):
        ax2.vlines(pos, 0, height, color=colors[1], linestyle="dashed", lw=2, alpha=0.6, label=labels[1])
    ax2.set_ylabel("sX-TDA oscillator strength", fontsize=15)
    ax2.tick_params(axis="both", which="major", labelsize=15)
    # ax2.set_ylim([0, maxy2])
    ax2.set_ylim([0, 2.1*maxy2])  # hhcrqpp2
    ax3.bar(ds2[0], ds2[1], width=1.5, label=ds2[3], color=ds2[2])

    ax3.set_xlim(min(x), max(x))
    ax3.set_xlabel("wavelength (nm)", fontsize=15)
    # ax.set_ylabel(r"$\epsilon$ / $M^{-1}cm^{-1}$", fontsize=15)
    if norm:
        ax1.set_ylabel(r"Normalized abs. intensity", fontsize=15)
    else:
        ax1.set_ylabel(r"$\epsilon$ / $M^{-1}cm^{-1}$", fontsize=15)
    ax1.tick_params(axis="both", which="major", labelsize=15)
    ax3.set_ylabel(r'$\Delta\langle \hat{S}^2\rangle$', fontsize=15)
    maxy = np.max((np.max(args[-1]), np.max(y)))
    # # print(maxy)
    # if not norm:
    #     if mol == 'ttm/':
    #         maxy = 41476.99925456449  # ttm fwhm=0.2
    #     elif mol == 'bispytm/':
    #         maxy = 29125.924840564476  # bispytm fwhm=0.2
    #     elif mol == 'ttm3ncz/':
    #         maxy = 62891.109187797156  # ttm3ncz fwhm=0.2
    #     elif mol == 'ptm3ncz/':
    #         maxy = 42615.05030815971  # ptm3ncz fwhm=0.2
    #     elif mol == 'mttm2/':
    #         maxy = 83331.98247503756  # mttm2 fwhm=0.2
    #     elif mol == 'hhcrqpp2/':
    #         maxy = 70018.44883471262  # hhcrqpp2 fwhm=0.2  row=300
    ax1.set_ylim([0, 1.1 * maxy])
    ax3.set_ylim(0, 2.1)

    # ax1.grid(True, linestyle="--", alpha=0.5)
    # ax2.grid(True, linestyle="--", alpha=0.5)
    # combine legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=15, loc='upper right')
    ax1.legend(fontsize=15)
    ax1.legend(framealpha=1)
    ax3.legend(framealpha=1)
    fig.tight_layout()

    # all_lines = lines1 + lines2
    # all_labels = labels1 + labels2
    # ax.legend(all_lines, all_labels,
    #       loc='lower center',
    #       bbox_to_anchor=(0.5, 1.02),
    #       ncol=np.ceil(nlabel/2),
    #       columnspacing=1.5,
    #       handlelength=2,
    #       fontsize=15)
    if mol == 'ttm/':
        ax1.set_xlim(210, 700)  # ttm, bispytm
        plt.axes([0.55, 0.5, 0.3, 0.3])  # ttm
    elif mol == 'bispytm/':
        ax1.set_xlim(210, 700)  # ttm, bispytm
        plt.axes([0.55, 0.55, 0.3, 0.3])  # bispytm
    elif mol == 'ttm3ncz/':
        ax1.set_xlim(270, 820)  # ttm3ncz
        plt.axes([0.35, 0.65, 0.35, 0.35])  # ttm3ncz
    elif mol == 'ptm3ncz/':
        ax1.set_xlim(260, 830)  # ptm3ncz
        plt.axes([0.5, 0.5, 0.35, 0.35])  # ptm3ncz
    elif mol == 'mttm2/':
        ax1.set_xlim(260, 620)  # mttm2
        plt.axes([0.58, 0.6, 0.28, 0.28])  # mttm2
    else:
        # ax1.set_xlim(205, 700)  # hhcrqpp2 pbe0-st-10eV
        ax1.set_xlim(240, 700)  # hhcrqpp2 tpssh-st-10eV
        plt.axes([0.6, 0.55, 0.25, 0.25])  # hhcrqpp2
    bgimg = plt.imread(file + mol + functional + mol[:-1] + '.bmp')
    plt.imshow(bgimg)
    plt.axis("off")
    plt.show()
    fig.savefig(title + '.eps', dpi=1200, bbox_inches='tight')
    fig.savefig(title + '.pdf', dpi=1200, bbox_inches='tight')


def UVvis3(*args, num_method, labels, colors, fwhm=0.3, title='UVspec', norm=False):
    '''this function plot different functional and different root figure,
    note, for XTDA change linestyle to '-' '''
    in_num = np.sum(num_method)
    assert in_num == len(labels)-1, "input data and label must have same length"
    minx = np.inf
    maxx = -np.inf
    for i in range(in_num+1):
        if min(args[2*i]) < minx:
            minx = min(args[2*i])
        if max(args[2*i]) > maxx:
            maxx = max(args[2*i])
    x = np.linspace(minx - 25, maxx + 25, 1000)  # nm as unit
    y = np.zeros((x.shape[0], in_num))

    def gaussian(x, x0, A, fwhm):
        wavenum = 1 / (unit.eVxnm / fwhm)
        return 1.3062974e8 * A / (1e7 * wavenum) * np.exp(-((1 / x - 1 / x0) / (1 * wavenum)) ** 2)

    for i in range(in_num):  # minor 1 is because last energy and oscillator is experiment data
        for pos, height in zip(args[2 * i], args[2 * i + 1]):
            y[:, i] += gaussian(x, pos, height, fwhm)
        if norm:
            y[:, i] = y[:, i] / np.max(y[:, i])

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    # ax2 = ax1.twinx()
    # experiment
    ax1.plot(args[-2], args[-1], color=colors[-1], lw=2, label=labels[-1])
    for i, l, c in zip(range(in_num), labels[:in_num], colors[:in_num]):
        ax1.plot(x, y[:, i], '--', color=c, lw=2, label=l)
    # for i, (pos, height) in enumerate(zip(args[0], args[1])):
    #     ax2.vlines(pos, 0, height, color=colors[1], linestyle="dashed", lw=2, alpha=0.6, label=labels[1])
    # ax2.set_ylabel("sX-TDA oscillator strength", fontsize=15)
    # ax2.tick_params(axis="both", which="major", labelsize=15)
    # ax2.set_ylim([0, maxy2])

    if mol == 'ttm/':
        ax1.set_xlim(210, 700)  # ttm, bispytm
    elif mol == 'bispytm/':
        ax1.set_xlim(210, 700)  # ttm, bispytm
    elif mol == 'ttm3ncz/':
        ax1.set_xlim(270, 820)  # ttm3ncz
    elif mol == 'ptm3ncz/':
        ax1.set_xlim(260, 830)  # ptm3ncz
    elif mol == 'mttm2/':
        ax1.set_xlim(260, 630)  # mttm2
    else:
        ax1.set_xlim(205, 700)  # hhcrqpp2 pbe0-st-10eV
        # ax1.set_xlim(240, 700)  # hhcrqpp2 tpssh-st-10eV

    if norm:
        ax1.set_ylabel(r"Normalized abs. intensity", fontsize=15)
    else:
        ax1.set_ylabel(r"$\epsilon$ / $M^{-1}cm^{-1}$", fontsize=15)
    ax1.set_xlabel(r"wave length / nm", fontsize=15)
    ax1.tick_params(axis="both", which="major", labelsize=15)
    maxy = np.max((np.max(args[-1]), np.max(y)))
    ax1.set_ylim([0, 1.1 * maxy])
    ax1.legend(fontsize=15)
    ax1.legend(framealpha=1)
    fig.tight_layout()
    plt.show()
    fig.savefig(title + '.eps', dpi=1200, bbox_inches='tight')
    fig.savefig(title + '.pdf', dpi=1200, bbox_inches='tight')


def UVvis4(*args, ds2, cv1, num_method, labels, colors, fwhm=0.3, title='UVspec', maxy2=1, norm=False):
    '''this function combine U and X in single figure'''
    in_num = np.sum(num_method)
    assert in_num == len(labels)-1, "input data and label must have same length"
    # for l in labels:
    #     if 'Exp.' in l:
    #         in_num -= 1
    minx = np.inf
    maxx = -np.inf
    for i in range(in_num+1):
        if min(args[2*i]) < minx:
            minx = min(args[2*i])
        if max(args[2*i]) > maxx:
            maxx = max(args[2*i])
    x = np.linspace(minx - 25, maxx + 25, 1000)  # nm as unit
    y = np.zeros((x.shape[0], in_num))

    def gaussian(x, x0, A, fwhm):
        wavenum = 1 / (unit.eVxnm / fwhm)
        return 1.3062974e8 * A / (1e7 * wavenum) * np.exp(-((1 / x - 1 / x0) / (1 * wavenum)) ** 2)

    for i in range(in_num):  # minor 1 is because last energy and oscillator is experiment data
        for pos, height in zip(args[2 * i], args[2 * i + 1]):
            y[:, i] += gaussian(x, pos, height, fwhm)
        if norm:
            y[:, i] = y[:, i] / np.max(y[:, i])

    fig, (ax1, ax3, ax4) = plt.subplots(
        3, 1, figsize=(8, 8), sharex=True,
        gridspec_kw={'height_ratios': [3, 1, 1], 'hspace':0.12}  # 高度比例：ax1最大
    )
    ax2 = ax1.twinx()
    # experiment
    ax1.plot(args[-2], args[-1], color=colors[-1], lw=2, label=labels[-1])
    # nlabel += 1
    # calculation
    for i, l, c in zip(range(in_num), labels[:in_num], colors[:in_num]):
        if mol == 'g3ttm/': # g3ttm
            if i in [0,2,4]:
                ax1.plot(x, y[:, i], '--', color=c, lw=2, label=l)
            if i in [1,3]:
                ax1.plot(x, y[:, i], color=c, lw=2, label=l)
        else:
            if i in [1,3,4]:
                ax1.plot(x, y[:, i], '--', color=c, lw=2, label=l)
            if i in [0,2]:
                ax1.plot(x, y[:, i], color=c, lw=2, label=l)
        # nlabel += 1
    # # peak position
    # for i in range(in_num):
    #     for j, (pos, height) in enumerate(zip(args[2*i], args[2*i+1])):
    #         ax2.vlines(pos, 0, height, color=colors[i], linestyle="dashed", alpha=0.6,
    #                    label=labels[i] if j == 0 else None)
    #     # nlabel += 1
    for i, (pos, height) in enumerate(zip(args[2], args[3])):
        ax2.vlines(pos, 0, height, color=colors[1], linestyle="dashed", lw=2, alpha=0.6, label=labels[1])
    ax2.set_ylabel("sX-TDA oscillator strength", fontsize=15)
    ax2.tick_params(axis="both", which="major", labelsize=15)
    ax2.set_ylim([0, maxy2])
    if mol == 'hhcrqpp2/':
        ax2.set_ylim([0, 2.1*maxy2])  # hhcrqpp2
    ax3.bar(ds2[0], ds2[1], width=1.5, label=ds2[3], color=ds2[2])
    ax4.bar(cv1[0], cv1[1], width=1.5, label=cv1[3], color=cv1[2])

    ax4.set_xlim(min(x), max(x))
    ax4.set_xlabel("wavelength (nm)", fontsize=15)
    # ax.set_ylabel(r"$\epsilon$ / $M^{-1}cm^{-1}$", fontsize=15)
    if norm:
        ax1.set_ylabel(r"Normalized abs. intensity", fontsize=15)
    else:
        ax1.set_ylabel(r"$\epsilon$ / $M^{-1}cm^{-1}$", fontsize=15)
    ax1.tick_params(axis="both", which="major", labelsize=15)
    ax3.set_ylabel(r'$\Delta\langle \hat{S}^2\rangle$', fontsize=15)
    ax4.set_ylabel('CV(1) ratio', fontsize=15)
    maxy = np.max((np.max(args[-1]), np.max(y)))
    # # print(maxy)
    # if not norm:
    #     if mol == 'ttm/':
    #         maxy = 41476.99925456449  # ttm fwhm=0.2
    #     elif mol == 'bispytm/':
    #         maxy = 29125.924840564476  # bispytm fwhm=0.2
    #     elif mol == 'ttm3ncz/':
    #         maxy = 62891.109187797156  # ttm3ncz fwhm=0.2
    #     elif mol == 'ptm3ncz/':
    #         maxy = 42615.05030815971  # ptm3ncz fwhm=0.2
    #     elif mol == 'mttm2/':
    #         maxy = 83331.98247503756  # mttm2 fwhm=0.2
    #     elif mol == 'hhcrqpp2/':
    #         maxy = 70018.44883471262  # hhcrqpp2 fwhm=0.2  row=300
    ax1.set_ylim([0, 1.1 * maxy])
    ax3.set_ylim(0, 2.1)
    ax4.set_ylim(0, 100)

    # ax1.grid(True, linestyle="--", alpha=0.5)
    # ax2.grid(True, linestyle="--", alpha=0.5)
    # combine legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=15, loc='upper right')
    ax1.legend(fontsize=15)
    ax1.legend(framealpha=1)
    ax3.legend(framealpha=1)
    ax4.legend(framealpha=1)
    # fig.tight_layout()

    # all_lines = lines1 + lines2
    # all_labels = labels1 + labels2
    # ax.legend(all_lines, all_labels,
    #       loc='lower center',
    #       bbox_to_anchor=(0.5, 1.02),
    #       ncol=np.ceil(nlabel/2),
    #       columnspacing=1.5,
    #       handlelength=2,
    #       fontsize=15)
    if mol == 'ttm/':
        ax1.set_xlim(210, 700)  # ttm, bispytm
        plt.axes([0.55, 0.45, 0.25, 0.25])  # ttm
    elif mol == 'bispytm/':
        ax1.set_xlim(210, 700)  # ttm, bispytm
        plt.axes([0.55, 0.45, 0.25, 0.25])  # bispytm
    elif mol == 'ttm3ncz/':
        ax1.set_xlim(270, 820)  # ttm3ncz
        plt.axes([0.35, 0.6, 0.3, 0.3])  # ttm3ncz
    elif mol == 'ptm3ncz/':
        ax1.set_xlim(260, 830)  # ptm3ncz
        plt.axes([0.35, 0.55, 0.3, 0.3])  # ptm3ncz
    elif mol == 'mttm2/':
        ax1.set_xlim(260, 630)  # mttm2
        plt.axes([0.645, 0.465, 0.25, 0.25])  # mttm2
    elif mol == 'hhcrqpp2/':
        # ax1.set_xlim(205, 700)  # hhcrqpp2 pbe0-st-10eV
        ax1.set_xlim(240, 700)  # hhcrqpp2 tpssh-st-10eV
        plt.axes([0.6, 0.48, 0.2, 0.2])  # hhcrqpp2
    else:
        ax1.set_xlim(267, 700)  # g3ttm pbe0-st-10eV
        plt.axes([0.6, 0.48, 0.24, 0.24])  # g3ttm
    bgimg = plt.imread(file + mol + functional + mol[:-1] + '.bmp')
    plt.imshow(bgimg)
    plt.axis("off")
    plt.show()
    fig.savefig(title + '.eps', dpi=1200, bbox_inches='tight')
    fig.savefig(title + '.pdf', dpi=1200, bbox_inches='tight')



# molpic = Image.open(file+mol+functional+mol[:-1]+'.bmp')
XsTDA_gsol = pd.read_csv(file+mol+functional+'XsTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
e8 = XsTDA_gsol[:rows, 0]
os8 = XsTDA_gsol[:rows, 1]
ds2_8 = XsTDA_gsol[:rows, 2]
# rs8 = XsTDA_gsol[:rows, 2]
# no use, XsTDA with solvent effect just add solvent in ground process, but XTDA and XTDDFT with solvent effect add solvent in excited process
# XTDDFT_sol = pd.read_csv(file+mol+solvent+'XTDDFT.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e10 = XTDDFT_sol[:, 0]
# os10 = XTDDFT_sol[:, 1]
# XTDA_sol = pd.read_csv(file+mol+functional+'XTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e11 = XTDA_sol[:rows, 0]
# os11 = XTDA_sol[:rows, 1]
# # rs11 = XTDA_gsol[:, 2]
UsTDA_gsol = pd.read_csv(file+mol+functional+'UsTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
e9 = UsTDA_gsol[:rows, 0]
os9 = UsTDA_gsol[:rows, 1]
ds2_9 = UsTDA_gsol[:rows, 2]
# # this result put in doctor paper, but do not put in this paper
# UsTDA_orca = pd.read_csv(file+mol+functional+'UsTDA'+'-ORCA'+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e13 = UsTDA_orca[:rows, 0]
# os13 = UsTDA_orca[:rows, 1]
# # no use, UsTDA with solvent effect just add solvent in ground process, but UTDA and UTDDFT with solvent effect add solvent in excited process
# UTDDFT_sol = pd.read_csv(file+mol+solvent+'UTDDFT.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e11 = UTDDFT_sol[:, 0]
# os11 = UTDDFT_sol[:, 1]
UTDA_sol = pd.read_csv(file+mol+functional+'UTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
e12 = UTDA_sol[:rows, 0]
os12 = UTDA_sol[:rows, 1]
experiment_sol = pd.read_csv(file+mol+functional+'experiment'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()  # one experiment data
# experiment_sol = pd.read_csv(file+mol+"various-sol-"+'experiment.csv', sep='[,\s]+', header=None, engine='python').to_numpy()  # various experiment data
e7 = experiment_sol[:,0]
if mol == 'ttm3ncz/':
    os7 = experiment_sol[:,3]
elif mol == 'ptm3ncz/':
    os7 = experiment_sol[:, 2]
elif mol == 'g3ttm/':
    os7 = experiment_sol[:,1]/np.max(experiment_sol[:,1])
else:
    os7 = experiment_sol[:,1]

int7 = interpolate.interp1d(e7,os7,kind='cubic')
e7 = np.linspace(min(e7), max(e7), 1000)
os7 = int7(e7)

# # include vacuum result
# maxy2 = np.max(np.concatenate((os0, os1, os2, os3, os4, os5, os8, os9))) * 1.1  # include TDDFT, TDA, sTDA
# maxy2 = np.max(np.concatenate((os0, os2, os5, os8, os9, os11, os12))) * 1.1  # include TDA, sTDA

# # solvent result and orca result
# maxy2 = np.max(np.concatenate((os8, os9, os11, os12, os13))) * 1.1

# only solvent result
# maxy2 = np.max(np.concatenate((os8, os9, os11, os12))) * 1.1
maxy2 = np.max(np.concatenate((os8, os9, os12))) * 1.1  # g3ttm

# # no use, this function divide X and U in two figure
# UVvis(
#     e8, os8, e11, os11, e9, os9, e12, os12, e13, os13, e7, os7, ds2=[e9, ds2_9, colors[2], 'sUTDA'], num_method=[2,3,1],
#     labels=['sXTDA', 'XTDA', 'sUTDA', 'UTDA', 'sUTDA orca', 'Exp.'],
#     # colors=[colors[8],colors[11],colors[9],colors[12], colors[13],colors[7]],
#     colors=colors,
#     title='UVspec', fwhm=0.2,
#     minx=minx, maxx=maxx, maxy2=maxy2, norm=True
# )


if mol == 'hhcrqpp2/':
    norm = False
else:
    norm = True
# # paper
# UVvis2(
#     e11, os11, e8, os8, e12, os12, e9, os9, e7, os7, ds2=[e9, ds2_9, colors[2], 'sUTDA'], num_method=[4],
#     labels=['X-TDA', 'sX-TDA', 'U-TDA', 'sU-TDA', 'Expt.'],
#     # colors=[colors[8],colors[11],colors[9],colors[12], colors[13],colors[7]],
#     colors=colors,
#     title='UVspec', fwhm=0.2, maxy2=maxy2, norm=norm
# )


# paper
functional = 'pbe0-st-10eV/'
XsTDA_gsol = pd.read_csv(file+mol+functional+'XsTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
e8 = XsTDA_gsol[:rows, 0]
os8 = XsTDA_gsol[:rows, 1]
ds2_8 = XsTDA_gsol[:rows, 2]
cv1_8 = XsTDA_gsol[:rows, -1]
UsTDA_gsol = pd.read_csv(file+mol+functional+'UsTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
e9 = UsTDA_gsol[:rows, 0]
os9 = UsTDA_gsol[:rows, 1]
ds2_9 = UsTDA_gsol[:rows, 2]
cv1_9 = UsTDA_gsol[:rows, -1]
# UVvis4(
#     e11, os11, e8, os8, e12, os12, e9, os9, e7, os7,
#     ds2=[e9, ds2_9, colors[2], 'sU-TDA'], cv1=[e8, cv1_8, colors[0], 'sX-TDA'], num_method=[4],
#     labels=['X-TDA', 'sX-TDA', 'U-TDA', 'sU-TDA', 'Expt.'],
#     colors=colors,
#     title='UVspec', fwhm=0.2, maxy2=maxy2, norm=norm
# )
# # g3ttm
UVvis4(
    e8, os8, e12, os12, e9, os9, e7, os7,
    ds2=[e9, ds2_9, colors[2], 'sU-TDA'], cv1=[e8, cv1_8, colors[0], 'sX-TDA'], num_method=[3],
    labels=['sX-TDA', 'U-TDA', 'sU-TDA', 'Expt.'],
    colors=["#EF2C2B", "#23B2E0", "#23B2E0", "#000000"],
    title='UVspec', fwhm=0.2, maxy2=maxy2, norm=norm
)

# # doctor paper
# UVvis2(
#     e11, os11, e8, os8, e12, os12, e9, os9, e13, os13, e7, os7, ds2=[e9, ds2_9, colors[2], 'sUTDA'], num_method=[5],
#     labels=['XTDA', 'sXTDA', 'UTDA', 'sUTDA', 'sUTDA orca', 'Exp.'],
#     # colors=[colors[8],colors[11],colors[9],colors[12], colors[13],colors[7]],
#     colors=colors_doc,
#     title='UVspec_doc', fwhm=0.2, maxy2=maxy2, norm=norm
# )


# # various molecule different functional by simplified method
# functional = 'tpssh-st-10eV/'
# UsTDA_gsol = pd.read_csv(file+mol+functional+'UsTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e14 = UsTDA_gsol[:rows, 0]
# os14 = UsTDA_gsol[:rows, 1]
# ds2_14 = UsTDA_gsol[:rows, 2]
# XsTDA_gsol = pd.read_csv(file+mol+functional+'XsTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e15 = XsTDA_gsol[:rows, 0]
# os15 = XsTDA_gsol[:rows, 1]
# ds2_15 = XsTDA_gsol[:rows, 2]
# functional = 'b3lyp-st/'
# UsTDA_gsol = pd.read_csv(file+mol+functional+'UsTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e16 = UsTDA_gsol[:rows, 0]
# os16 = UsTDA_gsol[:rows, 1]
# ds2_16 = UsTDA_gsol[:rows, 2]
# XsTDA_gsol = pd.read_csv(file+mol+functional+'XsTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e17 = XsTDA_gsol[:rows, 0]
# os17 = XsTDA_gsol[:rows, 1]
# ds2_17 = XsTDA_gsol[:rows, 2]
# functional = 'bhhlyp-st/'
# UsTDA_gsol = pd.read_csv(file+mol+functional+'UsTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e22 = UsTDA_gsol[:rows, 0]
# os22 = UsTDA_gsol[:rows, 1]
# ds2_22 = UsTDA_gsol[:rows, 2]
# XsTDA_gsol = pd.read_csv(file+mol+functional+'XsTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e23 = XsTDA_gsol[:rows, 0]
# os23 = XsTDA_gsol[:rows, 1]
# ds2_23 = XsTDA_gsol[:rows, 2]
# functional = 'pbe0-st/'
# UsTDA_gsol = pd.read_csv(file+mol+functional+'UsTDA'+solvent+'-t_P8eV.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e20 = UsTDA_gsol[:rows, 0]
# os20 = UsTDA_gsol[:rows, 1]
# ds2_20 = UsTDA_gsol[:rows, 2]
# XsTDA_gsol = pd.read_csv(file+mol+functional+'XsTDA'+solvent+'-t_P8eV.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e21 = XsTDA_gsol[:rows, 0]
# os21 = XsTDA_gsol[:rows, 1]
# ds2_21 = XsTDA_gsol[:rows, 2]
# # ******ttm3ncz test result*****
# functional = 'test/noDA-CSF/'
# XsTDA_gsol = pd.read_csv(file+mol+functional+'XsTDA'+solvent+'-noDA-CSF.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e46 = XsTDA_gsol[:rows, 0]
# os46 = XsTDA_gsol[:rows, 1]
# ds2_46 = XsTDA_gsol[:rows, 2]
# # *****ttm test result*****
# functional = 'test/param/'
# XTDA_gsol = pd.read_csv(file+mol+functional+'XsTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e36 = XTDA_gsol[:rows, 0]
# os36 = XTDA_gsol[:rows, 1]
# ds2_36 = XTDA_gsol[:rows, 2]
# functional = 'test/basis/'
# XTDA_gsol = pd.read_csv(file+mol+functional+'XTDA'+solvent+'-CCPVDZ.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e37 = XTDA_gsol[:rows, 0]
# os37 = XTDA_gsol[:rows, 1]
# ds2_37 = XTDA_gsol[:rows, 2]
# XTDA_gsol = pd.read_csv(file+mol+functional+'XTDA'+solvent+'-CCPVTZ.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e38 = XTDA_gsol[:rows, 0]
# os38 = XTDA_gsol[:rows, 1]
# ds2_38 = XTDA_gsol[:rows, 2]
# XTDA_gsol = pd.read_csv(file+mol+functional+'XTDA'+solvent+'-AUGCCPVTZ.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e44 = XTDA_gsol[:rows, 0]
# os44 = XTDA_gsol[:rows, 1]
# ds2_44 = XTDA_gsol[:rows, 2]
# functional = 'test/solmodel/'
# XTDA_gsol = pd.read_csv(file+mol+functional+'XTDA'+solvent+'-COSMO.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e39 = XTDA_gsol[:rows, 0]
# os39 = XTDA_gsol[:rows, 1]
# ds2_39 = XTDA_gsol[:rows, 2]
# XTDA_gsol = pd.read_csv(file+mol+functional+'XTDA'+solvent+'-IEFPCM.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e40 = XTDA_gsol[:rows, 0]
# os40 = XTDA_gsol[:rows, 1]
# ds2_40 = XTDA_gsol[:rows, 2]
# functional = 'test/excitsol/'
# XTDA_gsol = pd.read_csv(file+mol+functional+'XTDA'+solvent+'-EXCIT-SMD.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e41 = XTDA_gsol[:rows, 0]
# os41 = XTDA_gsol[:rows, 1]
# ds2_41 = XTDA_gsol[:rows, 2]
# functional = 'test/tddft/'
# XTDA_gsol = pd.read_csv(file+mol+functional+'XTDDFT'+solvent+'-B3LYP.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e42 = XTDA_gsol[:rows, 0]
# os42 = XTDA_gsol[:rows, 1]
# ds2_42 = XTDA_gsol[:rows, 2]
# XTDA_gsol = pd.read_csv(file+mol+functional+'XTDDFT'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e43 = XTDA_gsol[:rows, 0]
# os43 = XTDA_gsol[:rows, 1]
# ds2_43 = XTDA_gsol[:rows, 2]
# functional = 'test/correct/'
# XsTDA_gsol = pd.read_csv(file+mol+functional+'XsTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e45 = XsTDA_gsol[:rows, 0]
# os45 = XsTDA_gsol[:rows, 1]
# ds2_45 = XsTDA_gsol[:rows, 2]
# # # *****g3ttm test result*****
# XsTDA_gsol = pd.read_csv(file+mol+functional+'XsTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e47 = XsTDA_gsol[:rows, 0]
# os47 = XsTDA_gsol[:rows, 1]
# ds2_47 = XsTDA_gsol[:rows, 2]
# UVvis3(
#     e8, os8, e46, os46, e7, os7, num_method=[2],
#     labels=['sX-TDA', 'sX-TDA-noDA-CSF', 'Expt.'],
#     # colors=[colors[8],colors[11],colors[9],colors[12], colors[13],colors[7]],
#     colors=colors_functionals,
#     title='UVspec', fwhm=0.2, norm=norm
# )

# # ttm XTDA different functional
# functional = 'blyp-st-8eV/'  # before do not calculate blyp, so XTDA put in -8eV document
# XTDA = pd.read_csv(file+mol+functional+'XTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e30 = XTDA[:rows, 0]
# os30 = XTDA[:rows, 1]
# functional = 'tpssh-st-8eV/'  # before do not calculate tpssh, so XTDA put in -8eV document
# XTDA = pd.read_csv(file+mol+functional+'XTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e31 = XTDA[:rows, 0]
# os31 = XTDA[:rows, 1]
# functional = 'wb97xd-st/'
# XTDA = pd.read_csv(file+mol+functional+'XTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e32 = XTDA[:rows, 0]
# os32 = XTDA[:rows, 1]
# functional = 'b3lyp-st/'
# XTDA = pd.read_csv(file+mol+functional+'XTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e33 = XTDA[:rows, 0]
# os33 = XTDA[:rows, 1]
# functional = 'pbe0-st/'
# XTDA = pd.read_csv(file+mol+functional+'XTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e34 = XTDA[:rows, 0]
# os34 = XTDA[:rows, 1]
# functional = 'bhhlyp-st/'
# XTDA = pd.read_csv(file+mol+functional+'XTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e35 = XTDA[:rows, 0]
# os35 = XTDA[:rows, 1]
# UVvis3(
#     e30, os30, e31, os31, e32, os32, e33, os33, e34, os34, e35, os35, e7, os7, num_method=[6],
#     labels=['BLYP', 'TPSSH','WB97XD', 'B3LYP', 'PBE0', 'BHHLYP', 'Expt.'],
#     # colors=[colors[8],colors[11],colors[9],colors[12], colors[13],colors[7]],
#     colors=colors_functionals,
#     title='UVspec', fwhm=0.2, norm=norm
# )

# # C6H5NIT in ground solvent and full solvent
# functional = 'b3lyp/'
# XTDA_gsol = pd.read_csv(file+mol+functional+'XTDA'+'-GSOLV.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e18 = XTDA_gsol[:rows, 0]
# os18 = XTDA_gsol[:rows, 1]
# XTDA_sol = pd.read_csv(file+mol+functional+'XTDA'+'-SOLV.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e19 = XTDA_sol[:rows, 0]
# os19 = XTDA_sol[:rows, 1]
# UVvis3(
#     e18, os18, e19, os19, e7, os7, num_method=[2],
#     labels=['ground solv.','solv.', ''],
#     # colors=[colors[8],colors[11],colors[9],colors[12], colors[13],colors[7]],
#     colors=colors_functionals,
#     title='UVspec', fwhm=0.2, norm=norm
# )

# # mttm2 different root
# functional = 'nroot-b3lyp-st/'
# XsTDA_gsol = pd.read_csv(file+mol+functional+'XsTDA'+solvent+'-1root.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e24 = XsTDA_gsol[:rows,0]
# os24 = XsTDA_gsol[:rows,1]
# XsTDA_gsol = pd.read_csv(file+mol+functional+'XsTDA'+solvent+'-10root.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e25 = XsTDA_gsol[:rows,0]
# os25 = XsTDA_gsol[:rows,1]
# XsTDA_gsol = pd.read_csv(file+mol+functional+'XsTDA'+solvent+'-100root.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e26 = XsTDA_gsol[:rows,0]
# os26 = XsTDA_gsol[:rows,1]
# XsTDA_gsol = pd.read_csv(file+mol+functional+'XsTDA'+solvent+'-1000root.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e27 = XsTDA_gsol[:rows,0]
# os27 = XsTDA_gsol[:rows,1]
# XsTDA_gsol = pd.read_csv(file+mol+functional+'XsTDA'+solvent+'-10000root.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e28 = XsTDA_gsol[:rows,0]
# os28 = XsTDA_gsol[:rows,1]
# XsTDA_gsol = pd.read_csv(file+mol+functional+'XsTDA'+solvent+'-42928root.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
# e29 = XsTDA_gsol[:rows,0]
# os29 = XsTDA_gsol[:rows,1]
# UVvis3(
#     e24, os24, e25, os25, e26, os26, e27, os27, e28, os28, e29, os29, e7, os7, num_method=[6],
#     labels=['1 root','12 root','104 root','1043 root','10134 root','42928 root', 'Exptl.'],
#     # colors=[colors[8],colors[11],colors[9],colors[12], colors[13],colors[7]],
#     colors=colors_functionals,
#     title='UVspec', fwhm=0.2, norm=norm
# )