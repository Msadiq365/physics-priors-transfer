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
ACC, GREY, DARK = "#1f5fbf", "#9a9a9a", "#333333"
DSC = {"CWRU": "#1b9e77", "PAD": "#d95f02", "UORED": "#7570b3", "KAIST": "#e7298a", "HUST": "#66a61e"}
DSN = {"CWRU": "CWRU", "PAD": "Paderborn", "UORED": "UORED-VAFCLS", "KAIST": "KAIST", "HUST": "HUST"}
CLM = {"N": "o", "IR": "^", "OR": "s"}
os.makedirs("figures", exist_ok=True)


def save(fig, name):
    fig.savefig(f"figures/{name}.pdf"); fig.savefig(f"figures/{name}.png", dpi=600); plt.close(fig)


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
    for c, col, lw in (("N", GREY, 0.8), ("IR", ACC, 0.8), ("OR", "#c0392b", 0.8)):
        idx = [i for i, m in enumerate(meta) if m["dataset"] == d and T.usable(m) and m["cls"] == c and (d != "HUST" or m["geometry"] == "HUST_6205")]
        ax.plot(grid[lim], np.median(np.log1p(X[idx][:, lim]), 0), color=col, lw=lw, label={"N": "Healthy", "IR": "Inner race", "OR": "Outer race"}[c])
    for h in (1, 2, 3):
        for fo, col, ls in ((o["BPFO"], "#c0392b", ":"), (o["BPFI"], ACC, "--")):
            if h * fo <= 12:
                ax.axvline(h * fo, color=col, lw=0.6, ls=ls, alpha=0.9)
    ax.set_xlim(0, 12)
    ax.text(0.005, 1.02, DSN[d] + (" (type 6205)" if d == "HUST" else ""), transform=ax.transAxes, fontsize=7.5, fontweight="bold", va="bottom")
    ax.set_ylabel("log(1 + S)")
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
im = ax.imshow(M, cmap="Greys", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if np.isfinite(M[i, j]):
            best = M[i, j] >= np.nanmax(M[i]) - 1e-9
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6.3, color="white" if M[i, j] > 0.55 else DARK, fontweight="bold" if best else "normal")
            if best:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=ACC, lw=1.2))
ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms); ax.set_yticks(range(len(order))); ax.set_yticklabels([lab(k) + (" (confirmatory)" if k == "PAD2UORED" else "") for k in order])
ax.axhline(11.5, color=DARK, lw=0.8); ax.axvline(0.5, color=DARK, lw=0.8); ax.axvline(7.5, color=DARK, lw=0.5, ls=":")
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("Bearing-level macro-F1"); cb.outline.set_linewidth(0.5)
ax.set_title("Best arm in each setting outlined. C0 is train-free; WDCNN is exploratory.", loc="left", fontsize=7)
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
st = np.array([r[2] for r in rows]); es = np.array([r[5] for r in rows])
ok = np.abs(st - es) / st <= 0.02
jx = np.random.default_rng(1).uniform(-0.06, 0.06, len(st))
a1.scatter(st[ok] + jx[ok], es[ok], s=12, facecolor="white", edgecolor=DARK, lw=0.7, label="Within 2%", zorder=3)
a1.scatter(st[~ok] + jx[~ok], es[~ok], s=12, color=ACC, label="Outside 2%", zorder=3)
lims = [28, 37.2]; a1.plot(lims, lims, color=DARK, lw=0.6); a1.fill_between(lims, np.array(lims) * 0.98, np.array(lims) * 1.02, color="#eeeeee", lw=0)
a1.set_xlim(lims); a1.set_ylim(28, 32); a1.set_xlabel("Stored speed (Hz)"); a1.set_ylabel("Largest vibration peak within 10% (Hz)")
a1.set_title(f"(a) {ok.sum()} of {len(ok)} recordings within 2% (gate G3)", loc="left"); a1.legend(frameon=False, loc="upper right"); a1.text(36.9, 28.2, "grey band: within 2%", ha="right", fontsize=6.2, color=DARK)
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
