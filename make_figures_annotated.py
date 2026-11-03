"""Publication figures (Elsevier single column 90 mm, double column 190 mm). Output: figures/FigN.pdf and .png (600 dpi)."""
import json, os, sys, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, ".")
from ppt import transfer as T, baseline as B, signal as S, stats as ST
from ppt.data import load_cache
from ppt.loaders import sealed_manifest, load_sealed

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
MM = 1 / 25.4
ACC, GREY, DARK = "#0072B2", "#8c8c8c", "#222222"
ANN = "#B2182B"   # annotation colour (Okabe-Ito / ColorBrewer, colour-blind safe)
IRC, ORC = "#0072B2", "#D55E00"
def note(ax, xy, xytext, text, ha="left", **kw):
    ax.annotate(text, xy=xy, xytext=xytext, ha=ha, fontsize=6.6, color=ANN, fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), arrowprops=dict(arrowstyle="-|>", color=ANN, lw=0.9, shrinkA=1, shrinkB=2), **kw)
from matplotlib.patches import Ellipse
DSC = {"CWRU": "#009E73", "PAD": "#E69F00", "UORED": "#0072B2", "KAIST": "#CC79A7", "HUST": "#56B4E9"}
DSN = {"CWRU": "CWRU", "PAD": "Paderborn", "UORED": "UORED-VAFCLS", "KAIST": "KAIST", "HUST": "HUST"}
CLM = {"N": "o", "IR": "^", "OR": "s"}
os.makedirs("figures/annotated", exist_ok=True)


def save(fig, name):
    fig.savefig(f"figures/annotated/{name}.pdf"); fig.savefig(f"figures/annotated/{name}.png", dpi=600); plt.close(fig)


A = json.load(open("results/analysis_main.json"))
AS = json.load(open("results/analysis_sens_stored.json"))
cache = load_cache(); meta = cache["meta"]; X = cache["order"]

# ---------------- Fig 1: study design ----------------
fig, ax = plt.subplots(figsize=(190 * MM, 50 * MM)); ax.set_xlim(0, 100); ax.set_ylim(0, 27); ax.axis("off")
def box(x, y, w, h, title, lines, fc="#f2f2f2", ec=DARK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.5, fontweight="bold", color=DARK)
    ax.text(x + w / 2, y + h - 5.4, "\n".join(lines), ha="center", va="top", fontsize=6.1, color=DARK, linespacing=1.35)
def arrow(x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=DARK))
box(0.5, 4, 22, 21, "1. Development", ["CWRU and Paderborn only", "pipeline and recipe fixed", "gate G1 passed", "dry run of every arm", "MDD simulated"])
box(26.5, 4, 22, 21, "2. Freeze and register", ["code tag prereg-freeze", "checksums of all data", "OSF registration", "osf.io/qsh2a", "2 November 2026"], fc="#e6eefb", ec=ACC)
box(52.5, 4, 22, 21, "3. Open target data", ["UORED-VAFCLS, KAIST, HUST", "gates G2 to G4", "deviation D1:", "UORED shaft speed", "logged with its time"])
box(78.5, 4, 21, 21, "4. Report", ["H1 to H3: PAD to UORED", "12 directed pairs", "HUST leave-one-type-out", "controls, sensitivity,", "exploratory WDCNN"])
for x0, x1 in ((23.3, 26.2), (49.3, 52.2), (75.3, 78.2)):
    arrow(x0, x1, 14.5)
ax.text(50, 1.0, "Every physical bearing stays on one side of every split; the code asserts it in every run.", ha="center", fontsize=6.8, color=DARK, style="italic")
save(fig, "Fig1_design")

# ---------------- Fig 2: example envelope-order spectra ----------------
fig, axes = plt.subplots(5, 1, figsize=(190 * MM, 150 * MM), sharex=True)
grid = S.BASE_GRID; lim = grid <= 12
for ax, d in zip(axes, ["CWRU", "PAD", "UORED", "KAIST", "HUST"]):
    o = T.geo(next(m for m in meta if m["dataset"] == d and m.get("geometry", d) in (d, "HUST_6205")))
    if d == "HUST":
        o = T.orders_for("HUST_6205")
    for c, col, lw in (("N", GREY, 0.8), ("IR", IRC, 0.8), ("OR", ORC, 0.8)):
        idx = [i for i, m in enumerate(meta) if m["dataset"] == d and T.usable(m) and m["cls"] == c and (d != "HUST" or m["geometry"] == "HUST_6205")]
        ax.plot(grid[lim], np.median(np.log1p(X[idx][:, lim]), 0), color=col, lw=lw, label={"N": "Healthy", "IR": "Inner race", "OR": "Outer race"}[c])
    for h in (1, 2, 3):
        for fo, col, ls in ((o["BPFO"], ORC, ":"), (o["BPFI"], IRC, "--")):
            if h * fo <= 12:
                ax.axvline(h * fo, color=col, lw=0.6, ls=ls, alpha=0.9)
    ax.set_xlim(0, 12)
    ax.text(0.005, 1.02, DSN[d] + (" (type 6205)" if d == "HUST" else ""), transform=ax.transAxes, fontsize=7.5, fontweight="bold", va="bottom")
    ax.set_ylabel("log(1 + S)")
    ymax = ax.get_ylim()[1]
    if d == "PAD":
        for xo in (2.1, 4.0, 6.3, 8.4):
            ax.add_patch(Ellipse((xo, ymax * 0.82), 0.45, ymax * 0.32, fill=False, ec=ANN, lw=0.9))
        ax.text(11.0, ymax * 0.99, "healthy: 2.1, 4.0, 6.3, 8.4\norders, not fault orders", color=ANN, fontsize=6.2, fontweight="bold", ha="center", va="top", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    if d == "UORED":
        note(ax, (0.38, 1.55), (1.15, 2.15), "cage (FTF), 0.38")
        note(ax, (6.0, 2.6), (7.3, ymax * 0.95), "inner race: shaft-order\nharmonics 1x to 6x")
    if d == "KAIST":
        note(ax, (7.3, 3.0), (8.3, ymax * 0.93), "OR peaks 1 to 2% above\n2x and 3x BPFO (slip)")
    if d == "HUST":
        note(ax, (5.4, 3.3), (6.6, ymax * 0.93), "healthy unit also peaks\nnear BPFI (5.4, 10.8)")
axes[0].legend(loc="lower right", ncol=3, frameon=False, bbox_to_anchor=(1.0, 1.0))
axes[-1].set_xlabel("Order (multiples of shaft speed); dotted: BPFO harmonics, dashed: BPFI harmonics")
fig.align_ylabels(axes); fig.subplots_adjust(hspace=0.35); save(fig, "Fig2_spectra")

# ---------------- Fig 3: cross-rig analysis ----------------
z = np.load("results/crossrig.npz"); U, ds, cl, P = z["U"], z["ds"], z["cl"], z["P"]
Uc = U - U.mean(0); _, Sv, Vt = np.linalg.svd(Uc, full_matrices=False); pc = Uc @ Vt[:2].T; ev = Sv ** 2 / np.sum(Sv ** 2)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(190 * MM, 78 * MM))
for d in DSC:
    for c in CLM:
        mk = (ds == d) & (cl == c)
        if mk.any():
            a1.scatter(pc[mk, 0], pc[mk, 1], s=16, marker=CLM[c], facecolor=DSC[d] if c != "N" else "white", edgecolor=DSC[d], lw=0.8)
            a2.scatter(P[mk, 0], P[mk, 1], s=16, marker=CLM[c], facecolor=DSC[d] if c != "N" else "white", edgecolor=DSC[d], lw=0.8)
a1.set_xlabel(f"PC 1 ({ev[0]*100:.0f}% of variance)"); a1.set_ylabel(f"PC 2 ({ev[1]*100:.0f}%)")
a1.set_title("(a) Order spectrum (input of M2), unit means", loc="left")
a2.set_xlabel("C0 outer-race prominence, log P$_O$"); a2.set_ylabel("C0 inner-race prominence, log P$_I$")
a2.set_title("(b) C0 prominence scores, unit means", loc="left")
h1 = [plt.Line2D([], [], marker="o", ls="", color=DSC[d], label=DSN[d], ms=4) for d in DSC]
h2 = [plt.Line2D([], [], marker=CLM[c], ls="", mfc="white" if c == "N" else DARK, mec=DARK, label={"N": "Healthy (open)", "IR": "Inner race", "OR": "Outer race"}[c], ms=4) for c in CLM]
fig.legend(handles=h1 + h2, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.03))
hm = cl == "N"; cx, cy = np.median(P[hm, 0]), np.median(P[hm, 1])
a2.add_patch(Ellipse((cx, cy), 1.3, 1.0, fill=False, ec=ANN, lw=1.0, ls="--"))
a2.text(cx + 0.8, cy - 0.65, "healthy units of all five rigs", fontsize=6.4, color=ANN, fontweight="bold")
note(a1, (pc[(ds == "KAIST") & (cl == "IR"), 0].mean(), pc[(ds == "KAIST") & (cl == "IR"), 1].mean()), (-26, -14), "KAIST inner race:\nits own direction")
note(a1, (pc[(ds == "HUST") & (cl == "IR"), 0].min(), pc[(ds == "HUST") & (cl == "IR"), 1][np.argmin(pc[(ds == "HUST") & (cl == "IR"), 0])]), (-31, -2.5), "HUST inner race")
note(a1, (1.5, 0.5), (-12, -8.5), "most units overlap here;\nfault classes do not separate")
fig.subplots_adjust(bottom=0.2, wspace=0.28); save(fig, "Fig3_crossrig")

# ---------------- Fig 4: confirmatory per-bearing differences ----------------
fig, axes = plt.subplots(1, 3, figsize=(190 * MM, 62 * MM), sharey=True)
names = {"H1": "H1: M3 vs M2", "H2": "H2: M4 vs M5", "H3": "H3: M4 vs C0"}
rng = np.random.default_rng(3)
for ax, h in zip(axes, ("H1", "H2", "H3")):
    for res, x0, mk, lab in ((A, 0, "o", "Corrected speed (D1)"), (AS, 1, "s", "Stored speed (registered)")):
        t = res["confirmatory"][h]; d = np.array(t["d"]); jit = rng.uniform(-0.12, 0.12, len(d))
        ax.scatter(x0 + jit, d, s=12, marker=mk, facecolor="white", edgecolor=GREY, lw=0.7)
        lo, hi = t["ci_signflip"]; ax.errorbar(x0 + 0.3, t["mean_d"], yerr=[[t["mean_d"] - lo], [hi - t["mean_d"]]], fmt=mk, color=ACC if x0 == 0 else DARK, ms=4, capsize=2, lw=0.9)
        ax.text(x0 + 0.15, 0.86, f"p = {t['p']:.3f}\nHolm p = {t['p_holm']:.2f}", ha="center", va="bottom", fontsize=6.2)
    ax.axhline(0, color=DARK, lw=0.6); ax.set_xticks([0.15, 1.15]); ax.set_xticklabels(["Corrected\nspeed", "Stored\nspeed"]); ax.set_xlim(-0.4, 1.7)
    ax.set_title(names[h], loc="left"); ax.set_ylim(-0.75, 1.05)
    dd = np.array(A["confirmatory"][h]["d"]); win = {"H1": "M3", "H2": "M4", "H3": "M4"}[h]
    ax.text(0.15, 0.64, f"favour {win}: {int((dd > 0).sum())} of {len(dd)}", fontsize=6.2, color=ANN, fontweight="bold", ha="center")
    if h == "H3":
        ax.add_patch(Ellipse((0.0, -0.5), 0.42, 0.42, fill=False, ec=ANN, lw=0.9))
        ax.text(0.3, -0.42, "UORED 6 and 9\n(outer race):\nC0 better", fontsize=6.2, color=ANN, fontweight="bold", va="center")
axes[0].set_ylabel("Per-bearing Brier difference\n(positive favours the second-named arm)")
save(fig, "Fig4_confirmatory")

# ---------------- Fig 5: heatmap of macro-F1 ----------------
order = ["PAD2UORED", "UORED2PAD", "CWRU2KAIST", "KAIST2CWRU", "CWRU2PAD", "CWRU2UORED", "KAIST2PAD", "KAIST2UORED",
         "PAD2CWRU", "UORED2CWRU", "PAD2KAIST", "UORED2KAIST"] + [f"HUST_LOTO_{t}" for t in ("6204", "6205", "6206", "6207", "6208")]
arms = ["C0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "WDCNN"]
lab = lambda k: ("HUST, held-out " + k[-4:]) if k.startswith("HUST") else k.replace("2", " to ").replace("PAD", "Paderborn").replace("UORED", "UORED")
M = np.full((len(order), len(arms)), np.nan)
for i, k in enumerate(order):
    for j, a in enumerate(arms):
        if a in A["table"][k]:
            M[i, j] = A["table"][k][a]["macro_f1"]
fig, ax = plt.subplots(figsize=(150 * MM, 120 * MM))
im = ax.imshow(M, cmap="cividis", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if np.isfinite(M[i, j]):
            best = M[i, j] >= np.nanmax(M[i]) - 1e-9
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6.3, color=DARK if M[i, j] > 0.62 else "white", fontweight="bold" if best else "normal")
            if best:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=ANN, lw=1.4))
ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms); ax.set_yticks(range(len(order))); ax.set_yticklabels([lab(k) + (" (confirmatory)" if k == "PAD2UORED" else "") for k in order])
ax.axhline(11.5, color=DARK, lw=0.8); ax.axvline(0.5, color=DARK, lw=0.8); ax.axvline(7.5, color=DARK, lw=0.5, ls=":")
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.03, pad=0.06, aspect=40); cb.set_label("Bearing-level macro-F1"); cb.outline.set_linewidth(0.5)
ax.set_title("Best arm in each setting outlined in red. C0 is train-free; WDCNN is exploratory.", loc="left", fontsize=7)
for (r, c, t) in ((0, 6, "M6 beats C0 on the\nconfirmatory pair"), (8, 4, "M4 beats C0:\nPAD to CWRU"), (13, 8, "only WDCNN win")):
    ax.annotate(t, xy=(c + 0.45, r), xytext=(9.0, r + (0.9 if r == 13 else 0.2)), fontsize=6.2, color=ANN, fontweight="bold", va="center", arrowprops=dict(arrowstyle="-|>", color=ANN, lw=0.8), annotation_clip=False)
save(fig, "Fig5_heatmap")

# ---------------- Fig 6: M4 centre controls ----------------
fig, ax = plt.subplots(figsize=(150 * MM, 105 * MM))
for i, k in enumerate(order):
    v = A["table"][k]["M4"]; c = v["controls"]; y = len(order) - 1 - i
    lo, mean, hi = c["random"]
    ax.plot([lo, hi], [y, y], color=GREY, lw=4, alpha=0.45, solid_capstyle="butt"); ax.plot(mean, y, "o", color=GREY, ms=2.5)
    for p in ("pert-5", "pert-2", "pert+2", "pert+5"):
        ax.plot(c[p], y, "|", color=DARK, ms=7, mew=1.0)
    ax.plot(v["macro_f1"], y, "o", color=ACC, ms=5)
ax.set_yticks(range(len(order))); ax.set_yticklabels([lab(k) + (" (conf.)" if k == "PAD2UORED" else "") for k in order][::-1])
ax.set_xlim(-0.02, 1.0); ax.set_xlabel("Bearing-level macro-F1"); ax.grid(axis="x", color="#e5e5e5", lw=0.5); ax.set_axisbelow(True)
h = [plt.Line2D([], [], marker="o", ls="", color=ACC, label="True fault-order centres", ms=5),
     plt.Line2D([], [], color=GREY, lw=4, alpha=0.45, label="20 random centre sets (range, dot = mean)"),
     plt.Line2D([], [], marker="|", ls="", color=DARK, mew=1.0, ms=7, label="Centres scaled by 0.95, 0.98, 1.02, 1.05")]
ax.legend(handles=h, loc="lower center", frameon=False, ncol=3, bbox_to_anchor=(0.4, 1.0))
yy = lambda k: len(order) - 1 - order.index(k)
note(ax, (A["table"]["PAD2UORED"]["M4"]["controls"]["pert+2"], yy("PAD2UORED")), (0.74, yy("PAD2UORED") - 1.2), "+2% centres: 0.62\nvs 0.24 at true orders")
note(ax, (A["table"]["UORED2CWRU"]["M4"]["controls"]["pert+2"], yy("UORED2CWRU")), (0.72, yy("UORED2CWRU") - 1.6), "+2% centres: 0.66\nvs 0.00 at true orders")
note(ax, (A["table"]["PAD2CWRU"]["M4"]["macro_f1"], yy("PAD2CWRU")), (0.62, yy("PAD2CWRU") + 1.3), "only setting where true centres\nbeat every random set")
save(fig, "Fig6_centre_controls")

# ---------------- Fig 7: UORED speed evidence ----------------
rows = []
for e in sealed_manifest("data/sealed_export.npz"):
    if e["dataset"] != "UORED":
        continue
    x, fs, est = load_sealed(e); os.environ["UORED_SPEED"] = "stored"; _, _, st = load_sealed(e); del os.environ["UORED_SPEED"]
    Xf = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)))); f = np.fft.rfftfreq(len(x), 1 / 12000)
    bb = (f >= 0.9 * st) & (f <= 1.1 * st); pk = float(f[bb][np.argmax(Xf[bb])])
    rows.append((e["path"].split("__")[1], e["bearing_id"], st, est, x, pk))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(190 * MM, 66 * MM), gridspec_kw={"width_ratios": [1, 1.3]})
st = np.array([r[2] for r in rows]); es = np.array([r[3] for r in rows])
ok = np.abs(st - es) / st <= 0.02
jx = np.random.default_rng(1).uniform(-0.06, 0.06, len(st))
a1.scatter(st[ok] + jx[ok], es[ok], s=12, facecolor="white", edgecolor=DARK, lw=0.7, label="Within 2%", zorder=3)
a1.scatter(st[~ok] + jx[~ok], es[~ok], s=12, color=ACC, label="Outside 2%", zorder=3)
lims = [28, 37.2]; a1.plot(lims, lims, color=DARK, lw=0.6); a1.fill_between(lims, np.array(lims) * 0.98, np.array(lims) * 1.02, color="#eeeeee", lw=0)
a1.set_xlim(lims); a1.set_ylim(28, 32); a1.set_xlabel("Stored speed (Hz)"); a1.set_ylabel("Shaft speed from vibration spectrum (Hz)")
a1.set_title("(a) Stored vs measured shaft speed", loc="left")
note(a1, (32.45, 29.8), (30.6, 31.55), "bearing 8: stored 32.5 Hz")
note(a1, (36.52, 29.8), (34.0, 30.75), "bearing 7 (faulty):\nstored 36.5 Hz", ha="left")
a1.text(30.15, 31.75, "30 Hz synchronous", fontsize=6.2, color=ANN, rotation=90, va="top")
a1.axvline(30, color=ANN, lw=0.6, ls=":"); a1.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.08)); a1.text(36.9, 28.15, "grey band: within 2%", ha="right", fontsize=6.2, color=DARK)
r = next(r for r in rows if r[0] == "O_7_2"); x = r[4]
Xf = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)))); f = np.fft.rfftfreq(len(x), 1 / 12000); b = (f >= 20) & (f <= 80)
a2.plot(f[b], Xf[b] / Xf[b].max(), color=DARK, lw=0.6)
for v, col, txt in ((r[3], ACC, f"spectral estimate {r[3]:.1f} Hz"), (r[2], "#c0392b", f"stored {r[2]:.1f} Hz")):
    a2.axvline(v, color=col, lw=0.8, ls="--"); a2.axvline(2 * v, color=col, lw=0.6, ls=":")
    a2.text(v + 0.5, 0.95 if col == ACC else 0.8, txt, color=col, fontsize=6.5, va="top")
a2.set_xlabel("Frequency (Hz); dotted lines: second harmonic"); a2.set_ylabel("Normalised amplitude")
a2.set_title("(b) Recording O_7_2: raw vibration spectrum", loc="left")
save(fig, "Fig7_uored_speed")
print("figures written:", sorted(os.listdir("figures")))
