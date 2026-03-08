import shutil
from pathlib import Path


def build_3d_html(tmp_dir, outfile):
    """
    Copy HTML frames from tmp_dir into a new src_<viewer_stem> directory in cwd,
    then generate a timeline viewer.

    Output structure
        cwd/
            viewer.html
            viewer.src/
                timeline.js
                frame_0000.html
                frame_0001.html
                ...

    Parameters
    ----------
    tmp_dir : str or Path
        Directory containing per-frame HTML files.
    outfile : str
        Name of the main viewer HTML file written in cwd.

    Returns
    -------
    Path
        Path to the generated viewer HTML
    """

    parent_dir = Path(outfile).parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tmp_dir)

    if not tmp_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {tmp_dir}")

    frames = sorted([f for f in tmp_dir.glob("*.html") if f.is_file()])
    if not frames:
        raise ValueError(f"No HTML frame files found in: {tmp_dir}")

    src_dir_name = f"{Path(outfile).stem}.src"
    src_dir = parent_dir / src_dir_name

    if src_dir.exists():
        shutil.rmtree(src_dir)

    src_dir.mkdir(parents=True, exist_ok=True)

    frame_names = []
    for f in frames:
        dst = src_dir / f.name
        shutil.copy2(f, dst)
        frame_names.append(f.name)

    js = f"""
const frames = {frame_names};

const viewer = document.getElementById("viewer");
const lane = document.getElementById("timeline-lane");
const ticks = document.getElementById("ticks");
const thumb = document.getElementById("thumb");
const frameLabel = document.getElementById("frameLabel");

let current = 0;
let dragging = false;

function clamp(value, lo, hi) {{
    return Math.max(lo, Math.min(hi, value));
}}

function frameToPercent(i) {{
    if (frames.length <= 1) return 0;
    return (i / (frames.length - 1)) * 100;
}}

function percentToFrame(pct) {{
    if (frames.length <= 1) return 0;
    return Math.round((pct / 100) * (frames.length - 1));
}}

function loadFrame(i) {{
    current = clamp(i, 0, frames.length - 1);
    viewer.src = "{src_dir_name}/" + frames[current];
    updateUI();
}}

function updateUI() {{
    const pct = frameToPercent(current);
    thumb.style.left = pct + "%";
    frameLabel.style.left = pct + "%";
    frameLabel.textContent = String(current);
}}

function buildTicks() {{

    ticks.innerHTML = "";

    const maxLabels = 20;
    const step = Math.max(1, Math.ceil(frames.length / maxLabels));

    frames.forEach((_, i) => {{

        const tick = document.createElement("button");
        tick.type = "button";
        tick.className = "tick";
        tick.style.left = frameToPercent(i) + "%";

        let label = "";

        if (i % step === 0 || i === frames.length - 1) {{
            label = `<span class="tick-label">${{i}}</span>`;
        }}

        tick.innerHTML = `<span class="tick-mark"></span>${{label}}`;

        tick.addEventListener("click", () => loadFrame(i));

        ticks.appendChild(tick);

    }});
}}

function clientXToFrame(clientX) {{

    const rect = lane.getBoundingClientRect();
    const x = clamp(clientX - rect.left, 0, rect.width);
    const pct = rect.width === 0 ? 0 : (x / rect.width) * 100;

    return percentToFrame(pct);
}}

lane.addEventListener("click", (e) => {{

    if (e.target.closest(".tick")) return;

    loadFrame(clientXToFrame(e.clientX));
}});

thumb.addEventListener("mousedown", (e) => {{

    e.preventDefault();
    dragging = true;
}});

document.addEventListener("mousemove", (e) => {{

    if (!dragging) return;

    loadFrame(clientXToFrame(e.clientX));
}});

document.addEventListener("mouseup", () => {{
    dragging = false;
}});

function step(delta) {{
    loadFrame(current + delta);
}}

document.addEventListener("keydown", (e) => {{

    if (e.key === "ArrowLeft") step(-1);
    if (e.key === "ArrowRight") step(1);
}});

buildTicks();
loadFrame(0);
"""

    (src_dir / "timeline.js").write_text(js, encoding="utf-8")

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>

<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>3D Viewer</title>

<style>

:root {{
    --bg: #0b0d10;
    --panel: rgba(18,20,24,0.92);
    --panel-border: rgba(255,255,255,0.08);
    --track: #2b3138;
    --track-fill: #e7eaee;
    --text: #f5f7fa;
    --muted: #b7bec8;
    --button: #1a1f26;
    --button-hover: #232934;
    --lane-pad: 40px;
}}

* {{
    box-sizing: border-box;
}}

html, body {{
    margin: 0;
    width: 100%;
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: Inter, system-ui, sans-serif;
}}

body {{
    overflow: hidden;
}}

#viewer {{
    display: block;
    width: 100%;
    height: calc(100vh - 126px);
    border: 0;
    background: #000;
}}

#timeline {{
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    height: 126px;
    background: var(--panel);
    border-top: 1px solid var(--panel-border);
}}

#timeline-lane {{
    position: relative;
    height: 76px;
    margin: 0 var(--lane-pad);
}}

#ticks {{
    position: absolute;
    left: 0;
    right: 0;
    top: 8px;
    height: 34px;
}}

.tick {{
    position: absolute;
    top: 0;
    transform: translateX(-50%);
    border: 0;
    background: transparent;
    color: var(--muted);
    cursor: pointer;
    padding: 0;
    min-width: 28px;
}}

.tick-mark {{
    display: block;
    width: 2px;
    height: 16px;
    margin: 0 auto 4px auto;
    border-radius: 999px;
    background: rgba(255,255,255,0.45);
}}

.tick-label {{
    display: block;
    font-size: 11px;
    text-align: center;
    white-space: nowrap;
}}

#track {{
    position: absolute;
    left: 0;
    right: 0;
    top: 50px;
    height: 6px;
    border-radius: 999px;
    background: var(--track);
}}

#thumb {{
    position: absolute;
    top: 50px;
    width: 20px;
    height: 20px;
    transform: translate(-50%, -7px);
    border-radius: 50%;
    background: var(--track-fill);
    cursor: grab;
}}

#thumb:active {{
    cursor: grabbing;
}}

#frameLabel {{
    position: absolute;
    top: 0;
    transform: translateX(-50%);
    font-size: 14px;
    font-weight: 600;
}}

#controls {{
    display: flex;
    gap: 12px;
    align-items: center;
    padding: 10px var(--lane-pad);
}}

.timeline-btn {{
    border: 0;
    background: var(--button);
    color: var(--text);
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 14px;
    cursor: pointer;
}}

.timeline-btn:hover {{
    background: var(--button-hover);
}}

</style>

</head>

<body>

<iframe id="viewer"></iframe>

<div id="timeline">

    <div id="timeline-lane">

        <div id="ticks"></div>

        <div id="frameLabel">0</div>

        <div id="track"></div>

        <div id="thumb"></div>

    </div>

    <div id="controls">

        <button class="timeline-btn" onclick="step(-1)">◀ Prev</button>
        <button class="timeline-btn" onclick="step(1)">Next ▶</button>

    </div>

</div>

<script src="{src_dir_name}/timeline.js"></script>

</body>
</html>
"""

    viewer_path = parent_dir / outfile
    viewer_path.write_text(html, encoding="utf-8")

    return viewer_path
