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
mol = 'hhcrqpp2/'
if mol == 'hhcrqpp2/':
    rows = 300
else:
    rows = 100
# colors = ["#0B7055", "#796CAD", "#D65813", "#23BAC5", "#EECA40", "#FFA070",
#           "#000080", "#000000", "#6888F5", "#D77071", "#F0C284", "#EF8B67",
#           "#808080", "#85C3DC"]
# colors = ["#547bb4", "#629c35", "#6c61af", "#6f6f6f", "#c0321a", "#dd7c4f", "#000000"]
# colors = ["#EF2C2B", "#EEBEC0", "#23B2E0", "#A5CC5B", "#C2C2C2", "#000000"]
colors_doc = ["#EF2C2B", "#EF2C2B", "#23B2E0", "#23B2E0", "#A5CC5B", "#000000"]  # doctor paper
colors = ["#EF2C2B", "#EF2C2B", "#23B2E0", "#23B2E0", "#000000"]  # paper
# solvent = '-TOLUENE'
# solvent = '-CYCLOHEXANE'
solvent = '-ACETONITRILE'
functional = 'pbe0-st/'

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


def UVvis2(*args, ds2, num_method, labels, colors, fwhm=0.3, title='UVspec', minx=300, maxx=700, maxy2=1, norm=False):
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    # ax2 = ax.twinx()
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
    # # for i, (pos, height) in enumerate(zip(args[0], args[1])):
    # #     ax2.vlines(pos, 0, height, color=colors[0], linestyle="dashed", alpha=0.6,
    # #                label=label[0]+' $f$' if i == 0 else None)  # only for first method
    # ax2.set_ylabel("$f$", fontsize=15)
    # ax2.tick_params(axis="both", which="major", labelsize=15)
    # ax2.set_ylim([0, maxy2])
    ax2.bar(ds2[0], ds2[1], width=1.5, label=ds2[3], color=ds2[2])

    ax2.set_xlim(min(x), max(x))
    ax2.set_xlabel("wavelength (nm)", fontsize=15)
    # ax.set_ylabel(r"$\epsilon$ / $M^{-1}cm^{-1}$", fontsize=15)
    if norm:
        ax1.set_ylabel(r"Normalized abs. intensity", fontsize=15)
    else:
        ax1.set_ylabel(r"$\epsilon$ / $M^{-1}cm^{-1}$", fontsize=15)
    ax1.tick_params(axis="both", which="major", labelsize=15)
    ax2.set_ylabel(r'$\Delta\langle \hat{S}^2\rangle$', fontsize=15)
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
    ax2.set_ylim(0, 2.1)
    # ax1.grid(True, linestyle="--", alpha=0.5)
    # ax2.grid(True, linestyle="--", alpha=0.5)
    # # combine legend
    # lines1, labels1 = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax.legend(lines1 + lines2, labels1 + labels2, fontsize=15, loc='upper right')
    # ax.legend(fontsize=15)
    ax1.legend(framealpha=1)
    ax2.legend(framealpha=1)
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
        plt.axes([0.6, 0.5, 0.3, 0.3])  # ttm
    elif mol == 'bispytm/':
        plt.axes([0.65, 0.55, 0.3, 0.3])  # bispytm
    elif mol == 'ttm3ncz/':
        plt.axes([0.4, 0.7, 0.3, 0.3])  # ttm3ncz
    elif mol == 'ptm3ncz/':
        plt.axes([0.65, 0.55, 0.3, 0.3])  # ptm3ncz
    elif mol == 'mttm2/':
        plt.axes([0.7, 0.58, 0.25, 0.25])  # mttm2
    else:
        plt.axes([0.65, 0.55, 0.3, 0.3])  # hhcrqpp2
    bgimg = plt.imread(file + mol + functional + mol[:-1] + '.bmp')
    plt.imshow(bgimg)
    plt.axis("off")
    plt.show()
    # fig.savefig(title + '.eps', dpi=1200, bbox_inches='tight')
    # fig.savefig(title + '.pdf', dpi=1200, bbox_inches='tight')


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
XTDA_sol = pd.read_csv(file+mol+functional+'XTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
e11 = XTDA_sol[:rows, 0]
os11 = XTDA_sol[:rows, 1]
# rs11 = XTDA_gsol[:, 2]
UsTDA_gsol = pd.read_csv(file+mol+functional+'UsTDA'+solvent+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
e9 = UsTDA_gsol[:rows, 0]
os9 = UsTDA_gsol[:rows, 1]
ds2_9 = UsTDA_gsol[:rows, 2]
UsTDA_orca = pd.read_csv(file+mol+functional+'UsTDA'+'-ORCA'+'.csv', sep='[,\s]+', header=None, engine='python').to_numpy()
e13 = UsTDA_orca[:rows, 0]
os13 = UsTDA_orca[:rows, 1]
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
else:
    os7 = experiment_sol[:,1]

int7 = interpolate.interp1d(e7,os7,kind='cubic')
e7 = np.linspace(min(e7), max(e7), 1000)
os7 = int7(e7)

minx = np.inf
maxx = -np.inf
# for e in [e0, e2, e5, e7, e8, e9, e11, e12]:
# for e in [e8, e7, e11]:
for e in [e7, e8, e9, e11, e12, e13]:  # hhcrqpp2
    if min(e) < minx:
        minx = min(e)
    if max(e) > maxx:
        maxx = max(e)
# maxy2 = np.max(np.concatenate((os0, os1, os2, os3, os4, os5, os8, os9))) * 1.1  # include TDDFT, TDA, sTDA
# maxy2 = np.max(np.concatenate((os0, os2, os5, os8, os9, os11, os12))) * 1.1  # include TDA, sTDA
maxy2 = np.max(np.concatenate((os8, os9, os11, os12, os13))) * 1.1  # hhcrqpp2
# UVvis(
#     e8, os8, e11, os11, e9, os9, e12, os12, e13, os13, e7, os7, ds2=[e9, ds2_9, colors[2], 'sUTDA'], num_method=[2,3,1],
#     labels=['sXTDA', 'XTDA', 'sUTDA', 'UTDA', 'sUTDA orca', 'Exp.'],
#     # colors=[colors[8],colors[11],colors[9],colors[12], colors[13],colors[7]],
#     colors=colors,
#     title='UVspec', fwhm=0.2,
#     minx=minx, maxx=maxx, maxy2=maxy2, norm=True
# )

# paper
if mol == 'hhcrqpp2/':
    norm = False
else:
    norm = True
UVvis2(
    e11, os11, e8, os8, e12, os12, e9, os9, e7, os7, ds2=[e9, ds2_9, colors[2], 'sUTDA'], num_method=[4],
    labels=['X-TDA', 'sX-TDA', 'U-TDA', 'sU-TDA', 'Expt.'],
    # colors=[colors[8],colors[11],colors[9],colors[12], colors[13],colors[7]],
    colors=colors,
    title='UVspec', fwhm=0.2,
    minx=minx, maxx=maxx, maxy2=maxy2, norm=norm
)

# doctor paper
UVvis2(
    e11, os11, e8, os8, e12, os12, e9, os9, e13, os13, e7, os7, ds2=[e9, ds2_9, colors[2], 'sUTDA'], num_method=[5],
    labels=['XTDA', 'sXTDA', 'UTDA', 'sUTDA', 'sUTDA orca', 'Exp.'],
    # colors=[colors[8],colors[11],colors[9],colors[12], colors[13],colors[7]],
    colors=colors_doc,
    title='UVspec_doc', fwhm=0.2,
    minx=minx, maxx=maxx, maxy2=maxy2, norm=norm
)
