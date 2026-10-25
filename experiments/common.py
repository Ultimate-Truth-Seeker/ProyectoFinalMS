import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "figure.constrained_layout.use": True, "savefig.dpi": 160, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "font.size": 10, "axes.titlesize": 11,
    "axes.prop_cycle": plt.cycler(color=["#2a6f97", "#d1495b", "#edae49", "#00798c", "#66a182", "#8e5572", "#555555"]),
})


def guardar(fig, nombre):
    fig.savefig(os.path.join(FIG, nombre))
    plt.close(fig)


def volcar(nombre, d):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(os.path.join(RES, nombre), "w") as f:
        json.dump(d, f, indent=2, default=conv)
