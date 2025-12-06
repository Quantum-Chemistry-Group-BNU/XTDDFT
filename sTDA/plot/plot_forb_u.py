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
# mol = 'ttm/'
# mol = 'bispytm/'
# mol = 'ttm3ncz/'
# mol = 'ptm3ncz/'
# mol = 'mttm2/'
mol = 'hhcrqpp2/'
file = '../result/'

# ==== 参数 ====
# solvent = 'cyclohexane-'
# solvent = 'toluene-'
solvent = 'acetonitrile-'
csv_file = solvent+"UsTDAgsol-orb.csv"
img_dir = file+mol+functional+'cubeneed/'+'VCUBE/'
prefix = solvent+"UsTDA-" + mol[:-1]

# ==== 读入数据 ====
orbenergy = pd.read_csv(img_dir+csv_file, sep='[,\s]+', header=None, engine='python').to_numpy()
orbenergy = np.concatenate((np.expand_dims(orbenergy[:, 0], axis=1), orbenergy[:,1:]*unit.ha2eV), axis=1)

fig, ax = plt.subplots(figsize=(8,8))

orbfig_loc = np.linspace(np.min(orbenergy[:,1:]), np.max(orbenergy[:,1:]), orbenergy.shape[0])
nc = 2
no = 1
if mol == "mttm2/":
    no = 2
elif mol == 'hhcrqpp2/':
    no = 3
nv = 2
line_hlength = 0.3  # half line length
aloc = 0
bloc = 2
# ==== 画 α 和 β 轨道 ====
for j, o, spin, s in zip([aloc, bloc], [0,1], ["alpha", "beta"], ["α", "β"]):
    for i in range(orbenergy.shape[0]):
        energy = orbenergy[i][o+1]
        index = int(orbenergy[:, 0][i])

        # 能级线
        ax.hlines(energy, j-line_hlength, j+line_hlength, color="black", linewidth=1)
        if j == aloc and i<nc+no:
            ax.annotate(
                "",  # 不要文字
                xy=(j, energy+0.2),  # 箭头的尖端
                xytext=(j, energy-0.2),  # 箭头的起点
                arrowprops=dict(arrowstyle="->", color="black", lw=1)
            )
        elif j == bloc and i<nc:
            ax.annotate(
                "",  # 不要文字
                xy=(j, energy - 0.2),  # 箭头的尖端
                xytext=(j, energy + 0.2),  # 箭头的起点
                arrowprops=dict(arrowstyle="->", color="black", lw=1)
            )
        # 能级能量
        if j == 0:
            if i == 0:
                ax.text(j-1.0, energy-0.1, f"{energy:4.2f}", va="center", fontsize=15)
            elif i == 1:
                ax.text(j-1.0, energy+0.1, f"{energy:4.2f}", va="center", fontsize=15)
            elif i == 2:
                ax.text(j-1.0, energy, f"{energy:4.2f}", va="center", fontsize=15)
            elif i == 3:
                ax.text(j-1.0, energy-0.1, f"{energy:4.2f}", va="center", fontsize=15)
            elif i == 4:
                ax.text(j-1.0, energy+0.1, f"{energy:4.2f}", va="center", fontsize=15)
            else:
                ax.text(j-1.0, energy, f"{energy:4.2f}", va="center", fontsize=15)
        else:
            if i == 0:
                ax.text(j+0.4, energy-0.1, f"{energy:4.2f}", va="center", fontsize=15)
            elif i == 1:
                ax.text(j+0.4, energy+0.1, f"{energy:4.2f}", va="center", fontsize=15)
            elif i == 2:
                ax.text(j+0.4, energy, f"{energy:4.2f}", va="center", fontsize=15)
            elif i == 3:
                ax.text(j+0.4, energy-0.1, f"{energy:4.2f}", va="center", fontsize=15)
            elif i == 4:
                ax.text(j+0.4, energy+0.1, f"{energy:4.2f}", va="center", fontsize=15)
            elif i == 5:
                ax.text(j+0.4, energy+0.02, f"{energy:4.2f}", va="center", fontsize=15)
            else:
                ax.text(j+0.4, energy, f"{energy:4.2f}", va="center", fontsize=15)
        # 标签
        if j==0:
            ax.text(j-3.5, orbfig_loc[i], f"{index}{s}", va="center", fontsize=15)
        else:
            ax.text(j+2.9, orbfig_loc[i], f"{index}{s}", va="center", fontsize=15)
        # 插入轨道图像
        img_path = os.path.join(img_dir, f"{prefix}{index}{spin}.bmp")
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            imagebox = OffsetImage(img, zoom=0.19)
            if j == 0:
                ab = AnnotationBbox(imagebox, (j-1.91, orbfig_loc[i]), frameon=False)
            else:
                ab = AnnotationBbox(imagebox, (j+2.0, orbfig_loc[i]), frameon=False)
            ax.add_artist(ab)

# 添加连接线（按序号对应）
for i in [0, 1, 3, 4]:
    ax.plot([aloc+line_hlength, bloc - line_hlength],
            [orbenergy[i,1], orbenergy[i,2]], color='gray', linestyle='--', linewidth=1)
ax.text(1, np.sum(orbenergy[0, 1:])/2-0.22, "135(HOMO-1)", ha="center", fontsize=15)
ax.text(1, np.sum(orbenergy[1, 1:])/2+0.15, "136(HOMO)", ha="center", fontsize=15)
ax.text(0.4, np.sum(orbenergy[2, 1]), "137(SOMO)", va="center", fontsize=15)
ax.text(0.3, np.sum(orbenergy[2, 2]), "137(SUMO)", va="center", fontsize=15)
# ax.text(0.4, np.sum(orbenergy[3, 1])+0.1, "175(SOMO)", va="center", fontsize=15)
# ax.text(0.3, np.sum(orbenergy[3, 2])+0.1, "175(SUMO)", va="center", fontsize=15)
# ax.text(0.4, np.sum(orbenergy[4, 1])+0.05, "176(SOMO)", va="center", fontsize=15)
# ax.text(0.4, np.sum(orbenergy[4, 2])+0.1, "176(SUMO)", va="center", fontsize=15)
ax.text(1, np.sum(orbenergy[3, 1:])/2-0.2, "138(LOMO)", ha="center", fontsize=15)
ax.text(1, np.sum(orbenergy[4, 1:])/2+0.12, "139(LOMO+1)", ha="center", fontsize=15)
# # ax.text(1, np.sum(orbenergy[5, 1:])/2+0.15, "279(LOMO)", ha="center", fontsize=15)
# ax.text(0.5, np.sum(orbenergy[5, 1:])/2+0.2, "178(LOMO+1)", ha="center", fontsize=15)
# ax.text(0.5, np.sum(orbenergy[6, 1:])/2+0.2, "178(LOMO+1)", ha="center", fontsize=15)

# ==== 设置图形 ====
ax.axis("off")
ax.set_ylim(np.min(orbenergy[:, 1:])-0.5, np.max(orbenergy[:, 1:])+0.5)
ax.set_xlim(-2.5, 4.5)
# ax.set_ylabel("Energy (eV)")
ax.set_xticks([0,1])
# ax.set_xticklabels(["α", "β"])
ax.set_title("E/eV", fontsize=15)
plt.tight_layout()
plt.show()
fig.savefig("solvent-sUTDA-orb" + '.eps', dpi=600, bbox_inches='tight')
fig.savefig("solvent-sUTDA-orb" + '.pdf', dpi=600, bbox_inches='tight')
