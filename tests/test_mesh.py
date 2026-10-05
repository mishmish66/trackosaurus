import urllib.parse

import pytest

from trex import daemon, server
from trex.daemon import Roots
from trex.crawl import Crawl
from trex.index import Explorer

from helpers import get_json, post_json, request, wait_for, write_run


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(daemon, "PULL_EVERY", 0.1)


@pytest.fixture
def node(tmp_path, http_server):
    """make(name): a daemon's Roots named `name`, crawling a directory of one run named for it unless `crawls` is
    False, its server and its URL."""
    made: list[Roots] = []

    def make(name, crawls=True):
        roots = Roots(tmp_path / name / "cache", tmp_path / name / "state" / "roots.json", name=name)
        made.append(roots)
        if crawls:
            d = tmp_path / "files" / name / "runs"
            write_run(d / "r1")
            roots.add(d)
        srv = server.serve(None, "127.0.0.1", 0, roots)
        return roots, srv, http_server(srv)

    yield make
    for r in made:
        r.close()


def own(roots):
    return f"{roots.node.name}:{next(iter(roots.specs.values()))}"


def vias(roots, *nodes):
    """The holdings of `roots` with node ids written as the names of `nodes`."""
    names = {n.node.id: n.node.name for n in nodes}
    return {d["id"]: [names.get(v, v) for v in d["via"]] for d in roots.holdings()["dirs"]}


def synced(roots):
    """Whether every directory `roots` pulls holds its runs."""
    return all(len(roots.entries[d].runs("").runs) == 1 for d in list(roots.pulled))


def test_a_trex_pulls_every_directory_another_holds_and_names_them_alike(node):
    a, _, a_url = node("A")
    b, _, b_url = node("B", crawls=False)
    b.add_link(a_url)
    assert [r["name"] for r in b.served()] == [r["name"] for r in a.served()] == ["runs"]
    assert [(r["id"], r["link"]) for r in b.served()] == [(own(a), a_url)]
    assert vias(b, a, b) == {own(a): ["A", "B"]}
    assert wait_for(lambda: synced(b))
    assert get_json(f"{b_url}/api/runs")["runs"] == get_json(f"{a_url}/api/runs")["runs"]


def test_two_trex_pulling_from_each_other_hold_each_directory_once(node):
    (a, _, a_url), (b, _, b_url) = node("A"), node("B")
    a.add_link(b_url)
    b.add_link(a_url)
    want_a, want_b = {own(a): ["A"], own(b): ["B", "A"]}, {own(b): ["B"], own(a): ["A", "B"]}
    assert wait_for(lambda: vias(a, a, b) == want_a and vias(b, a, b) == want_b)
    for _ in range(5):  # rounds of pulling change nothing
        a._reconcile()
        b._reconcile()
    assert vias(a, a, b) == want_a and vias(b, a, b) == want_b and list(a.pulled) == [own(b)] and list(b.pulled) == [own(a)]


def test_three_trex_in_a_ring_each_hold_every_directory_once(node):
    (a, _, a_url), (b, _, b_url), (c, _, c_url) = node("A"), node("B"), node("C")
    a.add_link(b_url)
    b.add_link(c_url)
    c.add_link(a_url)
    ring = (a, b, c)
    assert wait_for(lambda: all(len(r.holdings()["dirs"]) == 3 for r in ring))
    assert {d: v[:-1] for d, v in vias(a, *ring).items()} == {own(a): [], own(b): ["B"], own(c): ["C", "B"]}
    assert {d: v[:-1] for d, v in vias(b, *ring).items()} == {own(b): [], own(c): ["C"], own(a): ["A", "C"]}
    assert {d: v[:-1] for d, v in vias(c, *ring).items()} == {own(c): [], own(a): ["A"], own(b): ["B", "A"]}
    assert wait_for(lambda: all(synced(r) for r in ring))


def test_a_directory_its_source_stops_crawling_leaves_a_cycle_that_holds_it(node):
    a, _, a_url = node("A")
    (b, _, b_url), (c, _, c_url) = node("B", crawls=False), node("C", crawls=False)
    b.add_link(a_url)
    c.add_link(b_url)
    b.add_link(c_url)
    gone = own(a)
    assert wait_for(lambda: gone in b.entries and gone in c.entries)
    a.remove("runs")
    assert wait_for(lambda: gone not in b.entries and gone not in c.entries)
    for _ in range(5):
        b._reconcile()
        c._reconcile()
    assert gone not in b.entries and gone not in c.entries


def test_an_unreachable_link_keeps_what_it_offered(node):
    a, a_srv, a_url = node("A")
    b, _, b_url = node("B", crawls=False)
    b.add_link(a_url)
    assert wait_for(lambda: synced(b))
    a_srv.shutdown()
    a_srv.server_close()
    a.close()
    assert wait_for(lambda: [(r["state"], r["link"]) for r in b.served()] == [("unreachable", a_url)])
    assert [r["id"] for r in get_json(f"{b_url}/api/runs")["runs"]] == ["runs/r1"]
    assert [link["state"] for link in get_json(f"{b_url}/api/daemon")["links"]] == ["unreachable"]


def test_a_directory_moves_to_another_link_when_its_link_is_removed(node):
    a, _, a_url = node("A")
    (b, _, b_url), (c, _, c_url) = node("B", crawls=False), node("C", crawls=False)
    b.add_link(a_url)
    c.add_link(a_url)
    c.add_link(b_url)
    assert wait_for(lambda: vias(c, a, b, c) == {own(a): ["A", "C"]} and synced(c))
    mirror = c.entries[own(a)]
    c.remove(a_url)
    assert wait_for(lambda: vias(c, a, b, c) == {own(a): ["A", "B", "C"]})
    assert c.entries[own(a)] is mirror and c.pulled[own(a)].link == b_url
    assert wait_for(lambda: mirror.origin.connected) and [r.id for r in mirror.runs("").runs] == ["r1"]


def test_links_are_saved_and_pulled_again_after_a_restart(node, tmp_path):
    a, _, a_url = node("A")
    b, _, _ = node("B", crawls=False)
    b.add_link(a_url)
    b.close()
    again = Roots(tmp_path / "B" / "cache", tmp_path / "B" / "state" / "roots.json")
    try:
        again.load()
        assert again.node == b.node and [link["url"] for link in again.links_info()] == [a_url]
        assert wait_for(lambda: list(again.pulled) == [own(a)] and synced(again))
    finally:
        again.close()


def test_links_are_added_listed_and_removed_over_http(node):
    a, _, a_url = node("A")
    b, _, b_url = node("B", crawls=False)
    assert post_json(f"{b_url}/api/daemon/add", {"path": a_url + "/"}) == (200, {"name": "", "url": "/"})
    info = get_json(f"{b_url}/api/daemon")
    assert info["node"] == {"id": b.node.id, "name": "B"}
    assert info["links"] == [{"url": a_url, "name": "A", "state": "connected", "error": ""}]
    assert [(r["name"], r["root"], r["link"]) for r in info["roots"]] == [("runs", own(a), a_url)]
    status, body = post_json(f"{b_url}/api/daemon/remove", {"name": "runs"})
    assert status == 400 and a_url in body["error"]
    assert post_json(f"{b_url}/api/daemon/remove", {"name": a_url}) == (200, {"ok": True})
    assert get_json(f"{b_url}/api/daemon")["roots"] == [] and b.history() == [a_url]
    assert post_json(f"{b_url}/api/daemon/add", {"path": "http://127.0.0.1:1"})[0] == 400


def test_a_trex_will_not_pull_from_itself(node):
    a, _, a_url = node("A")
    with pytest.raises(ValueError, match="is this trex"):
        a.add_link(a_url)


def test_directories_are_served_by_id_by_a_daemon_and_by_a_standalone_server(node, tmp_path, http_server):
    a, _, a_url = node("A")
    d = urllib.parse.quote(own(a), safe="")
    assert wait_for(lambda: [r["id"] for r in get_json(f"{a_url}/d/{d}/api/runs")["runs"]] == ["r1"])
    alone = Explorer(Crawl(tmp_path / "files" / "A" / "runs"), tmp_path / "alone-cache")
    alone.sync()
    url = http_server(server.serve(alone, "127.0.0.1", 0))
    holdings = get_json(f"{url}/api/holdings")
    ((only,),) = [[x["id"] for x in holdings["dirs"]]]
    assert holdings["node"] == server.STANDALONE._asdict() and only == server.standalone_id(alone)
    assert [r["id"] for r in get_json(f"{url}/d/{urllib.parse.quote(only, safe='')}/api/runs")["runs"]] == ["r1"]
    assert request(f"{url}/d/{urllib.parse.quote('elsewhere:/x', safe='')}/api/runs")[0] == 404
