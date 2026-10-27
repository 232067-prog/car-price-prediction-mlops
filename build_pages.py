from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = ROOT / "templates"
STATIC_DIR = ROOT / "static"
DIST_DIR = ROOT / "dist"


def clean_dist() -> None:
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True, exist_ok=True)


def build_index() -> None:
    template_path = TEMPLATES_DIR / "index.html"
    output_path = DIST_DIR / "index.html"

    html = template_path.read_text(encoding="utf-8")
    html = html.replace("{{ url_for('static', filename='style.css') }}", "static/style.css")
    html = html.replace("{{ url_for('static', filename='script.js') }}", "static/script.js")

    output_path.write_text(html, encoding="utf-8")


def copy_static_assets() -> None:
    target_static = DIST_DIR / "static"
    shutil.copytree(STATIC_DIR, target_static)


def main() -> None:
    clean_dist()
    build_index()
    copy_static_assets()
    print("Built static GitHub Pages site in ./dist")


if __name__ == "__main__":
    main()
