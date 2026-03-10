import json
import re
import shutil
from pathlib import Path

script_dir = Path(__file__).parent


def build_pvhtml(tmp_dir, outfile):
    tmp_dir, outfile = Path(tmp_dir), Path(outfile)
    parent_dir = outfile.parent
    template_path = script_dir / "data" / "pv"

    # 1. Setup Directories
    src_dir_name = f"{outfile.stem}.src"
    src_dir = parent_dir / src_dir_name
    data_dir = src_dir / "data"
    if src_dir.exists():
        shutil.rmtree(src_dir)
    src_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 2. Extract Data from PyVista Frames
    frames = sorted(f for f in tmp_dir.glob("*.html"))
    payload_re = re.compile(r'var\s+base64Str\s*=\s*"(?P<data>.*?)"\s*;', re.DOTALL)

    frame_files = []
    for i, frame in enumerate(frames):
        js_data = []
        payload = payload_re.search(frame.read_text()).group("data")
        js_name = f"idx{i}.js"
        js_data.append("window.__PV_FRAME_DATA__ = window.__PV_FRAME_DATA__ || {};")
        js_data.append(f'window.__PV_FRAME_DATA__[{i}] = "{payload}";')
        js_data.append("")

        with open(src_dir / "data" / js_name, "w", encoding="utf-8") as f:
            f.write("\n".join(js_data))
        frame_files.append(js_name)

    # 3. Load and Prep Templates
    js_template = (template_path / "timeline.js").read_text()
    ui_template = (template_path / "viewer.html").read_text()

    # Inject dynamic variables into JS
    final_js = js_template.replace("FRAMES_ARRAY", json.dumps(frame_files))
    final_js = final_js.replace("<<SRC_DIR>>", f"{src_dir_name}/data")
    (src_dir / "timeline.js").write_text(final_js)

    # 4. Assemble Final HTML
    # Use the first frame as the shell, but strip its unique data loader
    first_html = frames[0].read_text()
    shell_html = re.sub(
        r"<script>.*?OfflineLocalView\.load.*?</script>",
        "",
        first_html,
        flags=re.DOTALL,
    )

    # Inject the UI/CSS/JS block

    ui_block = ui_template.replace("<<SRC_DIR>>", src_dir_name)

    final_output = shell_html.replace("</body>", f"{ui_block}</body>")
    outfile.write_text(final_output)

    # copy timeline.css
    shutil.copy(template_path / "timeline.css", src_dir / "timeline.css")

    return outfile
