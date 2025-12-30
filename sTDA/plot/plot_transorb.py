import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import unit
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

functional = 'pbe0-st/'
# functional = 'tpssh-st-10eV/'
# mol = 'ttm/'
# mol = 'bispytm/'
# mol = 'ttm3ncz/'
# mol = 'ptm3ncz/'
# mol = 'mttm2/'
# mol = 'hhcrqpp2/'
mol = 'g3ttm/'
file = '../result/'

# ==== parameter ====
functional = 'pbe0-st-10eV-noFock-noDA/'  # G3TTM
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
# forb = [['136', '137'], ['137', '138'], ['135', '139'], ['135', '140'], ['136', '138']]
# excit = ['D1', 'D7', 'D60_1', 'D60_2', 'D60_3']

# # bispytm sUTDA
# forb = [['120beta', '121beta'], ['121alpha', '123alpha'],
#         ['120alpha', '129alpha'], ['120beta', '129beta']]
# excit = ['D1', 'D10', 'D83_1', 'D83_2']

# # bispytm sXTDA
# forb = [['120', '121'], ['121', '123'], ['115', '124'], ['117', '124'],
#         ['119', '126'], ['119', '130'], ['120', '129']]
# excit = ['D1', 'D10', 'D91_1', 'D91_2', 'D91_3', 'D91_4', 'D91_5']

# # # ttm3ncz sUTDA
# forb = [['204beta', '205beta'],
#         ['205alpha', '206alpha'],
#         ['204alpha', '206alpha'], ['205alpha', '207alpha'], ['204beta', '207beta']]

# # ttm3ncz sXTDA
# forb = [['204', '205'],
#         ['205', '206'], ['204', '206'],]
#         # ['203', '206'], ['189', '205'], ['190', '205']]
# excit = ['204-205', '205-206', '204-206']

# # mttm2 sUTDA
# forb = [['274beta', '276beta'], ['275beta', '277beta'],
#         ['276alpha', '278alpha'], ['277alpha', '279alpha'],]
# excit = ['274b-276b', '275b-277b', '276a-278a', '277a-279a']

# # mttm2 sXTDA
# forb = [['274', '277'], ['275', '276'],
#         ['276', '278'], ['277', '279'],]
#         # ['275', '278']]
# excit = ['274-277', '275-276', '276-278', '277-279']

# # hhcrqpp2 sUTDA
# forb = [['173beta', '174beta'],
#         ['173beta', '176beta'],
#         ['173alpha', '177alpha'],['173beta', '177beta'],]
# excit = ['173b-174b', '173b-176b', '173a-177a', '173b-177b']

# # hhcrqpp2 sXTDA pbe0-st-10eV
# forb = [['173', '177'],
#         ['173', '175'], ['173', '176'], ['173', '182'],
#         ['172', '178'],]
# excit = ['173-177', '173-175', '173-176', '173-182', '172-178']
# # hhcrqpp2 sXTDA tpssh-st-10eV
# forb = [['173', '177'], ['170', '175'], # ['173', '177'], have been exported, CV(1)
#         ['172', '178'], ['173', '185'], # ['173', '177'], have been exported
#         ['173', '181'],]
# excit = ['173-177', '170-175', '172-178', '173-185', '173-181']

# # ptm3ncz sUTDA
# forb = [['252beta', '253beta'], ['250alpha', '255alpha'], ['250beta', '255beta'], ['250alpha', '258alpha'],]
# excit = ['D1', 'D20', 'D92']

# # ptm3ncz sXTDA
# forb = [['252', '253'], ['253', '255'], ['250', '258'],]
# excit = ['D1', 'D18', 'D85']

# # g3ttm sUTDA
# forb = [['415alpha', '421alpha'], ['415beta', '421beta']]
# excit = ['415a-421a', '415b-421b']

# g3ttm sXTDA
forb = [['406', '419']]
excit = ['406-419']


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
