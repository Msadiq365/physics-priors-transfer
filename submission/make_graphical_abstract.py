import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
ACC, DARK, GREY = "#1f5fbf", "#333333", "#9a9a9a"
fig = plt.figure(figsize=(13 / 2.54 * 1.6, 5.2 / 2.54 * 1.6))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
def box(x, y, w, h, title, body, ec=DARK, fc="#f4f4f4"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h - 2.3, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=DARK)
    ax.text(x + w / 2, y + h - 7.2, body, ha="center", va="top", fontsize=7.8, color=DARK, linespacing=1.4)
box(1, 6, 28, 30, "Source rig", "labelled bearings\nenvelope-order spectra\n\nPaderborn, CWRU,\nUORED-VAFCLS, KAIST, HUST")
box(36, 6, 28, 30, "Eight arms, same pipeline", "train-free envelope baseline\n(prominence at fault orders)\n\n7 parameter-matched networks\nwith and without physics priors\n+ WDCNN, tuned variants")
box(71, 6, 28, 30, "Unseen target rig", "no target data in training\nno shared physical bearing\nanalysis registered in advance", ec=ACC, fc="#e8effb")
for x0, x1 in ((29.8, 35.4), (64.8, 70.4)):
    ax.annotate("", xy=(x1, 21), xytext=(x0, 21), arrowprops=dict(arrowstyle="-|>", lw=1.2, color=DARK))
ax.text(85, 17.5, "Envelope baseline best or tied\nin 14 of 17 settings", ha="center", fontsize=7.8, color=ACC, fontweight="bold", va="center")
ax.text(85, 12.2, "Physics priors: no gain", ha="center", fontsize=8.2, color=DARK)
ax.text(85, 7.8, "Healthy shift between rigs:\nspectrum 0.64, prominence 0.27", ha="center", fontsize=7.4, color=DARK, va="bottom")
ax.text(50, 1.5, "Physics priors and cross-rig transfer in bearing fault diagnosis", ha="center", fontsize=8.5, style="italic", color=DARK)
fig.savefig("submission/Graphical_abstract.pdf"); fig.savefig("submission/Graphical_abstract.png", dpi=300)
from PIL import Image; im = Image.open("submission/Graphical_abstract.png"); print(im.size)
