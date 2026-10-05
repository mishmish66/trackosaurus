import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import trex

BUILD = Path(__file__).resolve().parents[1] / "docs" / "build.py"


def test_docs_have_a_page_for_every_module_and_their_links_resolve(tmp_path: Path) -> None:
    subprocess.run([sys.executable, str(BUILD), str(tmp_path)], check=True, capture_output=True)
    names = {m.name for m in pkgutil.iter_modules(trex.__path__)} - {"__main__"}
    assert {p.stem for p in (tmp_path / "trex").glob("*.html")} == names
    assert "trex systemd-unit" in (tmp_path / "trex" / "node.html").read_text()
    assert '<video src="media/tour.mp4"' in (tmp_path / "trex.html").read_text() and (tmp_path / "media" / "tour.mp4").is_file()
    for page in tmp_path.rglob("*.html"):
        for href in re.findall(r'href="([^"#:]+\.html)', page.read_text()):
            assert (page.parent / href).resolve().is_file(), f"{page.name} links to missing {href}"
