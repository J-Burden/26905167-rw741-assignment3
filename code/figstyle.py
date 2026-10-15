import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for f in glob.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf"):
    fm.fontManager.addfont(f)

COL = 3.5     # single column width in inches
FULL = 7.16   # full text width in inches
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

C = {"knn": "#0072B2", "tree": "#D55E00", "bagging": "#009E73",
     "rf": "#CC79A7", "grey": "#7f7f7f", "ceiling": "#000000"}
LABEL = {"knn": "Nearest neighbours", "tree": "Classification tree",
         "bagging": "Bagged trees", "rf": "Random forest"}
