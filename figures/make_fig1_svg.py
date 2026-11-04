"""Figure 1 (study workflow) as hand-built SVG. Text colour and border colour are deliberately distinct:
text is near-black (#1A1A1A) or white on header bars; borders and arrows use stage colours."""
W, H = 1600, 960
TXT, SUB = "#1A1A1A", "#4A4A4A"
STAGES = [  # (x, width, border colour, light fill, title, subtitle)
    (20, 470, "#1F5F8B", "#EEF4F9", "Stage 1: Development", "CWRU and Paderborn only; target data sealed"),
    (520, 300, "#2E7D32", "#EEF6EE", "Stage 2: Preregistration", "Protocol, code and checksums frozen"),
    (850, 340, "#B26A00", "#FBF3E6", "Stage 3: Target data opened", "Gates checked before any confirmatory run"),
    (1220, 360, "#6A3D9A", "#F3EEF8", "Stage 4: Analysis and reporting", "Every registered run reported"),
]
out = []
a = out.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif">')
a('<defs>')
for i, (_, _, c, *_r) in enumerate(STAGES):
    a(f'<marker id="ar{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
a('<marker id="arK" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#555555"/></marker>')
a('</defs>')
a(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')


def text(x, y, s, size=17, weight="normal", fill=TXT, anchor="middle", style=""):
    a(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {style}>{s}</text>')


def lines(x, y, rows, size=16, gap=22, fill=TXT, anchor="middle", weight="normal"):
    for k, r in enumerate(rows):
        text(x, y + k * gap, r, size, weight, fill, anchor)


def box(x, y, w, h, stroke, fill="#FFFFFF", r=10, sw=2, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def arrow(x1, y1, x2, y2, i, sw=2.5):
    m = "arK" if i == "K" else f"ar{i}"
    col = "#555555" if i == "K" else STAGES[i][2]
    a(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{sw}" marker-end="url(#{m})"/>')


def database(cx, cy, c, label):
    w, h, e = 70, 66, 11
    a(f'<path d="M{cx-w/2},{cy-h/2+e} v{h-2*e} a{w/2},{e} 0 0 0 {w},0 v-{h-2*e}" fill="#FFFFFF" stroke="{c}" stroke-width="2.2"/>')
    a(f'<ellipse cx="{cx}" cy="{cy-h/2+e}" rx="{w/2}" ry="{e}" fill="#FFFFFF" stroke="{c}" stroke-width="2.2"/>')
    for k in (1, 2):
        a(f'<path d="M{cx-w/2},{cy-h/2+e+k*15} a{w/2},{e} 0 0 0 {w},0" fill="none" stroke="{c}" stroke-width="1.4"/>')
    text(cx, cy + h / 2 + 22, label, 16, "bold")


def check_shield(cx, cy, c):
    a(f'<path d="M{cx},{cy-34} l28,10 v18 c0,18 -12,30 -28,38 c-16,-8 -28,-20 -28,-38 v-18 z" fill="#FFFFFF" stroke="{c}" stroke-width="2.5"/>')
    a(f'<path d="M{cx-12},{cy+2} l9,10 l17,-20" fill="none" stroke="{c}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>')


def lock(cx, cy, c):
    a(f'<rect x="{cx-18}" y="{cy-4}" width="36" height="28" rx="4" fill="#FFFFFF" stroke="{c}" stroke-width="2.5"/>')
    a(f'<path d="M{cx-11},{cy-4} v-9 a11,11 0 0 1 22,0 v9" fill="none" stroke="{c}" stroke-width="2.5"/>')
    a(f'<circle cx="{cx}" cy="{cy+10}" r="3.5" fill="{c}"/>')


def gauge(cx, cy, c):
    a(f'<path d="M{cx-26},{cy+8} a26,26 0 0 1 52,0" fill="none" stroke="{c}" stroke-width="3"/>')
    a(f'<line x1="{cx}" y1="{cy+8}" x2="{cx+14}" y2="{cy-10}" stroke="{c}" stroke-width="3" stroke-linecap="round"/>')
    a(f'<circle cx="{cx}" cy="{cy+8}" r="4" fill="{c}"/>')


def spectrum_icon(x, y, c):
    a(f'<rect x="{x}" y="{y}" width="64" height="40" rx="4" fill="#FFFFFF" stroke="{c}" stroke-width="1.8"/>')
    pts = [(4, 34), (10, 30), (14, 12), (18, 30), (26, 28), (30, 8), (34, 30), (42, 27), (46, 16), (50, 30), (60, 31)]
    a('<polyline points="' + " ".join(f"{x+px},{y+py}" for px, py in pts) + f'" fill="none" stroke="{c}" stroke-width="1.8"/>')


# ---- stage frames and header bars ----
TOP, BOT = 20, 870
for i, (x, w, c, f, title, sub) in enumerate(STAGES):
    box(x, TOP, w, BOT - TOP, c, f, 14, 2.5)
    a(f'<rect x="{x}" y="{TOP}" width="{w}" height="70" rx="14" fill="{c}"/>')
    a(f'<rect x="{x}" y="{TOP+50}" width="{w}" height="20" fill="{c}"/>')
    text(x + w / 2, TOP + 32, title, 21, "bold", "#FFFFFF")
    text(x + w / 2, TOP + 57, sub, 14.5, "normal", "#FFFFFF")

# ---- stage 1 ----
c1 = STAGES[0][2]
database(80, 170, c1, "CWRU")
database(80, 300, c1, "Paderborn")
box(160, 115, 310, 125, c1)
text(315, 143, "Pipeline built and fixed", 17, "bold")
lines(315, 168, ["envelope-order spectra, 7 network arms", "and a train-free baseline (C0)", "gate G1 passed on both datasets"], 15, 21)
box(160, 260, 310, 105, c1)
text(315, 288, "Dry run and power analysis", 17, "bold")
lines(315, 313, ["every arm on CWRU to Paderborn and back", "minimum detectable difference simulated"], 15, 21)
arrow(118, 175, 158, 175, 0); arrow(118, 300, 158, 305, 0)
box(40, 400, 430, 215, c1, "#FFFFFF", 10, 2, "7,5")
text(255, 428, "Target datasets downloaded, kept sealed", 17, "bold")
for k, (nm, dx) in enumerate((("UORED-VAFCLS", 120), ("KAIST", 255), ("HUST", 390))):
    database(dx, 490, c1, nm)
lines(255, 573, ["label-blind export to one file, SHA-256 checksum", "read only: manifest, array lengths, UORED speed column"], 14.5, 20)
box(40, 640, 430, 200, c1)
text(255, 668, "Development log (DEV_LOG.md)", 17, "bold")
lines(70, 700, ["every recipe change dated before registration", "the speed-column read declared in advance", "G1 history and dry-run results kept", "for reporting in full"], 15, 24, anchor="start")
arrow(255, 365, 255, 398, 0)

# ---- stage 2 ----
c2 = STAGES[1][2]
cx2 = 670
check_shield(cx2 - 50, 160, c2); lock(cx2 + 50, 160, c2)
box(540, 220, 260, 250, c2)
text(cx2, 250, "Frozen before opening", 17, "bold")
lines(560, 282, ["hypotheses H1 to H3", "metric: per-bearing Brier score", "test: exact sign-flip, Holm", "quality gates G2 to G4", "code: commit 331fdff", "tag: prereg-freeze", "data: SHA-256 checksums"], 15, 25, anchor="start")
box(540, 495, 260, 120, c2, "#FFFFFF")
text(cx2, 527, "OSF registration", 18, "bold")
text(cx2, 557, "osf.io/qsh2a", 18, "bold", c2)
text(cx2, 585, "2 November 2026", 15)
box(540, 640, 260, 200, c2)
text(cx2, 668, "Update after opening", 17, "bold")
lines(560, 700, ["deviation D1 declared", "two fold-handling fixes", "gate results posted", "exploratory additions", "listed separately"], 15, 25, anchor="start")
arrow(cx2, 470, cx2, 493, 1); arrow(cx2, 615, cx2, 638, 1)

# ---- stage 3 ----
c3 = STAGES[2][2]
for k, (nm, y) in enumerate((("UORED-VAFCLS", 120), ("KAIST", 215), ("HUST", 310))):
    box(870, y, 140, 75, c3)
    spectrum_icon(878, y + 8, c3)
    text(980, y + 33, "", 1)
    text(940, y + 66, nm, 14.5, "bold")
box(1040, 120, 130, 265, c3)
gauge(1105, 165, c3)
text(1105, 215, "Gates", 17, "bold")
lines(1105, 243, ["G2: diagnosable", "by envelope?", "G3: isolation,", "checksums,", "speed check", "G4: shuffled", "labels"], 13.5, 19)
for y in (157, 252, 347):
    arrow(1010, y, 1038, y, 2)
box(870, 420, 300, 190, c3)
text(1020, 450, "UORED shaft-speed check (G3)", 17, "bold")
lines(1020, 478, ["stored speed matched the vibration", "spectrum in only 21 of 40 recordings", "stop and correct: label-blind", "spectral rule registered for KAIST"], 14.5, 22)
box(870, 640, 300, 200, c3, "#FFFFFF", 10, 2.5)
text(1020, 670, "Deviation D1 logged", 17, "bold")
lines(1020, 700, ["corrected speed used in main analysis", "stored speed kept as sensitivity run", "envelope check supports correction", "(16 of 18 differing recordings)", "conclusion unchanged under both"], 14.5, 24)
arrow(1105, 385, 1105, 418, 2); arrow(1020, 610, 1020, 638, 2)

# ---- stage 4 ----
c4 = STAGES[3][2]
box(1240, 110, 320, 205, c4)
text(1400, 140, "Confirmatory (registered)", 17, "bold")
lines(1400, 168, ["Paderborn to UORED-VAFCLS", "10 physical test bearings"], 15, 21)
for k, (h, t) in enumerate((("H1", "M3 vs M2"), ("H2", "M4 vs M5"), ("H3", "M4 vs C0"))):
    y = 222 + k * 30
    a(f'<rect x="1268" y="{y-19}" width="46" height="26" rx="5" fill="{c4}"/>')
    text(1291, y, h, 15, "bold", "#FFFFFF")
    text(1330, y, t, 15, "normal", TXT, "start")
    text(1532, y, "n.s.", 14, "bold", c4, "end")
box(1240, 340, 320, 140, c4)
text(1400, 370, "Descriptive", 17, "bold")
lines(1400, 398, ["12 directed transfer pairs", "HUST leave-one-type-out (5 folds)", "17 settings, every arm reported"], 15, 23)
box(1240, 505, 320, 160, c4)
text(1400, 535, "Controls and sensitivity", 17, "bold")
lines(1400, 563, ["shifted and random fault-order centres", "shuffled-label control", "stored UORED speed", "model-free rig and fault analysis"], 15, 23)
box(1240, 690, 320, 150, c4, "#FFFFFF", 10, 2, "7,5")
text(1400, 720, "Exploratory (not registered)", 17, "bold")
lines(1400, 748, ["WDCNN raw-signal network", "per-arm hyperparameter search", "labelled exploratory;", "no hypothesis test"], 15, 23)
arrow(1400, 315, 1400, 338, 3); arrow(1400, 480, 1400, 503, 3); arrow(1400, 665, 1400, 688, 3)

# ---- arrows between stages ----
for x1, x2, y in ((490, 520, 445), (820, 850, 445), (1190, 1220, 445)):
    a(f'<line x1="{x1-2}" y1="{y}" x2="{x2+2}" y2="{y}" stroke="#555555" stroke-width="5" marker-end="url(#arK)"/>')

# ---- bottom band ----
box(20, 892, W - 40, 50, "#555555", "#F2F2F2", 10, 2)
text(W / 2, 924, "Throughout: every physical bearing stays on one side of every split, and the code asserts this in every run.", 18, "bold")
a('</svg>')
open("/tmp/claude-0/-home-claude/4fbfedd7-5e16-5370-8fe9-e9214d6c4fe1/scratchpad/fig1/Figure_1_workflow.svg", "w").write("\n".join(out))
print("svg written")
