import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import unit
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

functional = 'pbe0-st/'
# mol = 'ttm/'
# mol = 'bispytm/'
# mol = 'ttm3ncz/'
# mol = 'ptm3ncz/'
mol = 'mttm2/'
# mol = 'hhcrqpp2/'
file = '../result/'

# ==== parameter ====
img_dir = file+mol+functional+'cubeneed/'+'VCUBE/'
# method = 'sUTDA-'
method = 'sXTDA-'
# solvent = 'cyclohexane-'
solvent = 'toluene-'
# solvent = 'acetonitrile-'
prefix = solvent + method[1]+method[0]+method[2:] + mol[:-1]

# ==== frontier orbitals ====
# # ttm sUTDA
# forb = [['136beta', '137beta'], ['137alpha', '139alpha'], ['137alpha', '147alpha']]
# excit = ['D1', 'D8', 'D96']

# # ttm sXTDA
# forb = [['136', '137'], ['137', '139'], ['135', '139'], ['135', '140'], ['136', '138']]
# excit = ['D1', 'D8', 'D60_1', 'D60_2', 'D60_3']

# # bispytm sUTDA
# forb = [['120beta', '121beta'], ['121alpha', '123alpha'],
#         ['120alpha', '129alpha'], ['120beta', '129beta']]
# excit = ['D1', 'D10', 'D83_1', 'D83_2']

# # bispytm sXTDA
# forb = [['120', '121'], ['121', '123'], ['115', '124'], ['117', '124'],
#         ['119', '126'], ['119', '130'], ['120', '129']]
# excit = ['D1', 'D10', 'D91_1', 'D91_2', 'D91_3', 'D91_4', 'D91_5']

# # ttm3ncz sUTDA
# forb = [['204beta', '205beta'], ['205alpha', '206alpha'],
#         ['204alpha', '210alpha'], ['205alpha', '214alpha'], ['203beta', '207beta']]
# excit = ['D1', 'D9', 'D37_1', 'D37_2', 'D37_3']

# # ttm3ncz sXTDA
# forb = [['204', '205'], ['205', '206'], ['205', '207'], ['204', '207'],
#         ['203', '206'], ['189', '205'], ['190', '205']]
# excit = ['D1', 'D10_1', 'D10_2', 'D10_3', 'D47_1', 'D47_2', 'D47_3']

# # mttm2 sUTDA
# forb = [['274beta', '276beta'], ['275beta', '277beta'],
#         ['276alpha', '278alpha'], ['277alpha', '279alpha'],
#         ['274alpha', '278alpha'], ['275alpha', '279alpha']]
# excit = ['D1_1', 'D1_2', 'D13_1', 'D13_2', 'D87_1', 'D87_2']

# mttm2 sXTDA
forb = [['274', '277'], ['275', '276'], ['276', '278'], ['277', '279'],
        ['275', '278']]
excit = ['D1_1', 'D1_2', 'D13_1', 'D13_2', 'D98']

# # hhcrqpp2 sUTDA
# forb = [['173beta', '174beta'],
#         ['174alpha', '178alpha'], ['175alpha', '179alpha'],
#         ['158beta', '175beta'], ['160beta', '175beta']]
# excit = ['D1', 'D21_1', 'D21_2', 'D257_1', 'D257_2']

# # hhcrqpp2 sXTDA
# forb = [['172', '176'], ['171', '177'], ['172', '177'],
#         ['172', '178'], ['172', '179'],
#         ['169', '189'], ['169', '189']]
# excit = ['D1_1', 'D1_2', 'D1_3', 'D18_1', 'D18_2', 'D265_1', 'D265_2']

# # ptm3ncz sUTDA
# forb = [['252beta', '253beta'], ['250alpha', '255alpha'], ['250beta', '255beta'], ['250alpha', '258alpha'],]
# excit = ['D1', 'D20', 'D92']

# # ptm3ncz sXTDA
# forb = [['252', '253'], ['253', '255'], ['250', '258'],]
# excit = ['D1', 'D18', 'D85']


# for i, t, info, j in zip(forb, title, information, range(len(title))):
for i, j in zip(forb, excit):
    fig, ax = plt.subplots(figsize=(8, 3))
    img_path1 = os.path.join(img_dir, f"{prefix}{i[0]}.bmp")
    img_path2 = os.path.join(img_dir, f"{prefix}{i[1]}.bmp")
    if os.path.exists(img_path1) and os.path.exists(img_path2):
        img1 = plt.imread(img_path1)
        imagebox1 = OffsetImage(img1, zoom=0.4)
        ab1 = AnnotationBbox(imagebox1, (0, 0), frameon=False)
        ax.add_artist(ab1)
        img2 = plt.imread(img_path2)
        imagebox2 = OffsetImage(img2, zoom=0.4)
        ab2 = AnnotationBbox(imagebox2, (1, 0), frameon=False)
        ax.add_artist(ab2)
    # ax.text(0.3, 0.9, t, ha="center", fontsize=15)
    # ax.text(0.3, -1, info, ha="center", fontsize=15)
    ax.annotate(
        "",  # 不要文字
        xy=(0.58, 0),  # 箭头的尖端
        xytext=(0.38, 0),  # 箭头的起点
        arrowprops=dict(arrowstyle="->", color="black", lw=1)
    )
    ax.set_ylim(-1, 1)
    ax.set_xlim(-0.5, 1.5)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
    fig.savefig("solvent-"+method+'transorb'+f'{j}'+'.eps', dpi=600, bbox_inches='tight')
    fig.savefig("solvent-"+method+'transorb'+f'{j}'+'.pdf', dpi=600, bbox_inches='tight')
