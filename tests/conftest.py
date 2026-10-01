import pytest

from trex import index


@pytest.fixture(autouse=True)
def close_explorers(monkeypatch):
    """Close every Explorer a test opens."""
    opened = []
    init = index.Explorer.__init__

    def tracked(self, *a, **kw):
        init(self, *a, **kw)
        opened.append(self)

    monkeypatch.setattr(index.Explorer, "__init__", tracked)
    yield
    for ex in opened:
        if not ex._closed:
            ex.close()
