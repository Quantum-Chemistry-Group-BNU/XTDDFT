import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import unit
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# run this code must finish plot orbital figure and store these figure in 'cubeneed' in each molecule

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

functional = 'pbe0-st/'
# functional = 'tpssh-st-10eV/'
# mol = 'ttm/'
# mol = 'bispytm/'
# mol = 'ttm3ncz/'
mol = 'ptm3ncz/'
# mol = 'mttm2/'
# mol = 'hhcrqpp2/'
file = '../result/'

# ==== 参数 ====
# solvent = 'cyclohexane-'
solvent = 'toluene-'
# solvent = 'acetonitrile-'
csv_file = solvent+"XsTDAgsol-orb.csv"
# img_dir = os.path.expanduser("~/master/xtddft/sTDA/result/bispytm/pbe0-st/cubeneed/VCUBE")
img_dir = file+mol+functional+'cubeneed/'+'VCUBE/'
prefix = solvent+"XsTDA-" + mol[:-1]

# ==== 读入数据 ====
orbenergy = pd.read_csv(img_dir+csv_file, sep='[,\s]+', header=None, engine='python').to_numpy()
orbenergy = np.concatenate((np.expand_dims(orbenergy[:, 0], axis=1), orbenergy[:,1:]*unit.ha2eV), axis=1)

fig, ax = plt.subplots(figsize=(8,8))

orbfig_loc = np.linspace(np.min(orbenergy[:,1:]), np.max(orbenergy[:,1:]), orbenergy.shape[0])
nc = 2
no = 1
if mol == "mttm2/":
    no = 2
elif mol == "hhcrqpp2/":
    no = 3
nv = 2
line_hlength = 0.3  # half line length
aloc = 0
# ==== 画 α 和 β 轨道 ====
for i in range(orbenergy.shape[0]):
    energy = orbenergy[i][1]
    index = int(orbenergy[:, 0][i])

    # 能级线
    ax.hlines(energy, -line_hlength, line_hlength, color="black", linewidth=1)
    if i < nc+no:
        ax.annotate(
            "",  # 不要文字
            xy=(0-0.02, energy+0.5),  # 箭头的尖端
            xytext=(0-0.02, energy-0.5),  # 箭头的起点
            arrowprops=dict(arrowstyle="->", color="black", lw=1)
        )
    if i < nc:
        ax.annotate(
            "",  # 不要文字
            xy=(0+0.02, energy-0.5),  # 箭头的尖端
            xytext=(0+0.02, energy+0.5),  # 箭头的起点
            arrowprops=dict(arrowstyle="->", color="black", lw=1)
        )
    # if i == 1:
    #     ax.annotate(
    #         "",  # 不要文字
    #         xy=(0 + 0.02, energy - 0.4),  # 箭头的尖端
    #         xytext=(0 + 0.02, energy + 0.6),  # 箭头的起点
    #         arrowprops=dict(arrowstyle="->", color="black", lw=1)
    #     )
    #     ax.annotate(
    #         "",  # 不要文字
    #         xy=(0 - 0.02, energy + 0.6),  # 箭头的尖端
    #         xytext=(0 - 0.02, energy - 0.4),  # 箭头的起点
    #         arrowprops=dict(arrowstyle="->", color="black", lw=1)
    #     )
    # 能级能量
    if i == 0:
        ax.text(-0.7, energy-0.25, f"{energy:4.2f}", va="center", fontsize=15)
    elif i == 1:
        ax.text(-0.7, energy+0.25, f"{energy:4.2f}", va="center", fontsize=15)
    elif i == 2:
        ax.text(-0.7, energy-0.5, f"{energy:4.2f}", va="center", fontsize=15)
    elif i == 3:
        ax.text(-0.7, energy, f"{energy:4.2f}", va="center", fontsize=15)
    elif i == 4:
        ax.text(-0.7, energy+0.5, f"{energy:4.2f}", va="center", fontsize=15)
    elif i == 5:
        ax.text(-0.7, energy-0.25, f"{energy:4.2f}", va="center", fontsize=15)
    else:
        ax.text(-0.7, energy+0.25, f"{energy:4.2f}", va="center", fontsize=15)
    # 标签
    ax.text(-1.9, orbfig_loc[i], f"{index}", va="center", fontsize=15)
    # 插入轨道图像
    img_path = os.path.join(img_dir, f"{prefix}{index}.bmp")
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        imagebox = OffsetImage(img, zoom=0.2)
        ab = AnnotationBbox(imagebox, (-1.2, orbfig_loc[i]), frameon=False)
        ax.add_artist(ab)

ax.text(0.4, np.sum(orbenergy[0, 1])-0.25, "172(HOMO-1)", va="center", fontsize=15)
ax.text(0.4, np.sum(orbenergy[1, 1])+0.25, "173(HOMO)", va="center", fontsize=15)
ax.text(0.4, np.sum(orbenergy[2, 1])-0.5, "174(SOMO)", va="center", fontsize=15)
ax.text(0.4, np.sum(orbenergy[3, 1]), "175(SOMO)", va="center", fontsize=15)
ax.text(0.4, np.sum(orbenergy[4, 1])+0.5, "176(SOMO)", va="center", fontsize=15)
ax.text(0.4, np.sum(orbenergy[5, 1])-0.25, "177(LOMO)", va="center", fontsize=15)
ax.text(0.4, np.sum(orbenergy[6, 1])+0.25, "178(LOMO+1)", va="center", fontsize=15)

# ==== 设置图形 ====
ax.axis("off")
ax.set_ylim(np.min(orbenergy[:, 1:])-0.5, np.max(orbenergy[:, 1:])+0.5)
ax.set_xlim(-2., 2.)
# ax.set_ylabel("Energy (eV)")
ax.set_xticks([0,1])
# ax.set_xticklabels(["α", "β"])
ax.set_title("E/eV", fontsize=15)
plt.tight_layout()
plt.show()
fig.savefig("solvent-sXTDA-orb" + '.eps', dpi=600, bbox_inches='tight')
fig.savefig("solvent-sXTDA-orb" + '.pdf', dpi=600, bbox_inches='tight')
