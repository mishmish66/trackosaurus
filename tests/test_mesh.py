import urllib.parse
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol

import pytest

from trex import node as trex_node, server
from trex.crawl import Crawl
from trex.index import Explorer
from trex.mirror import Pull
from trex.node import Node, dir_base
from trex.server import Server

from helpers import get_json, node_of, post_json, request, wait_for, write_run

type Made = tuple[Node, Server, str]


class Make(Protocol):
    def __call__(self, name: str, crawls: bool = True) -> Made: ...


@pytest.fixture(autouse=True)
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trex_node, "PULL_EVERY", 0.1)


@pytest.fixture
def make(tmp_path: Path, http_server: Callable[[Server], str]) -> Iterator[Make]:
    """make(name): a node named `name` that keeps what it holds, crawling a directory of one run named for it unless
    `crawls` is False, its server and its URL."""
    made: list[Node] = []

    def one(name: str, crawls: bool = True) -> Made:
        n = Node(tmp_path / name / "cache", tmp_path / name / "state" / "roots.json", name=name)
        made.append(n)
        if crawls:
            d = tmp_path / "files" / name / "runs"
            write_run(d / "r1")
            n.add(d)
        srv = server.serve(n, "127.0.0.1", 0)
        return n, srv, http_server(srv)

    yield one
    for n in made:
        n.close()


def own(n: Node) -> str:
    """The id of the directory node `n` crawls."""
    (d,) = n.crawled
    return d


def vias(n: Node, *nodes: Node) -> dict[str, list[str]]:
    """The holdings of `n` with node ids written as the names of `nodes`."""
    names = {x.identity.id: x.identity.name for x in nodes}
    return {o.id: [names.get(v, v) for v in o.via] for o in n.holdings().dirs}


def synced(n: Node) -> bool:
    """Whether every directory `n` pulls holds its runs."""
    return all(len(n.entries[d].runs("").runs) == 1 for d in list(n.pulled))


def test_a_trex_pulls_every_directory_another_holds_and_names_them_alike(make: Make) -> None:
    a, _, a_url = make("A")
    b, _, b_url = make("B", crawls=False)
    b.add_link(a_url)
    assert [d.name for d in b.served()] == [d.name for d in a.served()] == ["runs"]
    assert [(d.id, d.link) for d in b.served()] == [(own(a), a_url)]
    assert vias(b, a, b) == {own(a): ["A", "B"]}
    assert wait_for(lambda: synced(b))
    assert get_json(f"{b_url}/api/runs")["runs"] == get_json(f"{a_url}/api/runs")["runs"]


def test_two_trex_pulling_from_each_other_hold_each_directory_once(make: Make) -> None:
    (a, _, a_url), (b, _, b_url) = make("A"), make("B")
    a.add_link(b_url)
    b.add_link(a_url)
    want_a, want_b = {own(a): ["A"], own(b): ["B", "A"]}, {own(b): ["B"], own(a): ["A", "B"]}
    assert wait_for(lambda: vias(a, a, b) == want_a and vias(b, a, b) == want_b)
    for _ in range(5):  # rounds of pulling change nothing
        a.reconcile()
        b.reconcile()
    assert vias(a, a, b) == want_a and vias(b, a, b) == want_b and list(a.pulled) == [own(b)] and list(b.pulled) == [own(a)]


def test_three_trex_in_a_ring_each_hold_every_directory_once(make: Make) -> None:
    (a, _, a_url), (b, _, b_url), (c, _, c_url) = make("A"), make("B"), make("C")
    a.add_link(b_url)
    b.add_link(c_url)
    c.add_link(a_url)
    ring = (a, b, c)
    assert wait_for(lambda: all(len(r.holdings().dirs) == 3 for r in ring))
    assert {d: v[:-1] for d, v in vias(a, *ring).items()} == {own(a): [], own(b): ["B"], own(c): ["C", "B"]}
    assert {d: v[:-1] for d, v in vias(b, *ring).items()} == {own(b): [], own(c): ["C"], own(a): ["A", "C"]}
    assert {d: v[:-1] for d, v in vias(c, *ring).items()} == {own(c): [], own(a): ["A"], own(b): ["B", "A"]}
    assert wait_for(lambda: all(synced(r) for r in ring))


def test_a_directory_its_source_stops_crawling_leaves_a_cycle_that_holds_it(make: Make) -> None:
    a, _, a_url = make("A")
    (b, _, b_url), (c, _, c_url) = make("B", crawls=False), make("C", crawls=False)
    b.add_link(a_url)
    c.add_link(b_url)
    b.add_link(c_url)
    gone = own(a)
    assert wait_for(lambda: gone in b.entries and gone in c.entries)
    a.remove("runs")
    assert wait_for(lambda: gone not in b.entries and gone not in c.entries)
    for _ in range(5):
        b.reconcile()
        c.reconcile()
    assert gone not in b.entries and gone not in c.entries


def test_an_unreachable_link_keeps_what_it_offered(make: Make) -> None:
    a, a_srv, a_url = make("A")
    b, _, b_url = make("B", crawls=False)
    b.add_link(a_url)
    assert wait_for(lambda: synced(b))
    a_srv.shutdown()
    a_srv.server_close()
    a.close()
    assert wait_for(lambda: [(d.state, d.link) for d in b.served()] == [("unreachable", a_url)])
    assert [r["id"] for r in get_json(f"{b_url}/api/runs")["runs"]] == ["runs/r1"]
    assert [link["state"] for link in get_json(f"{b_url}/api/node")["links"]] == ["unreachable"]


def test_a_restarted_trex_answers_for_what_it_pulled_before_its_link_does(make: Make, tmp_path: Path) -> None:
    a, a_srv, a_url = make("A")
    b, _, _ = make("B", crawls=False)
    b.add_link(a_url)
    assert wait_for(lambda: synced(b))
    d = own(a)
    b.close()
    a_srv.shutdown()
    a_srv.server_close()
    a.close()
    again = Node(tmp_path / "B" / "cache", tmp_path / "B" / "state" / "roots.json")
    again.load()
    try:
        assert again.identity == b.identity and [link.url for link in again.links_info()] == [a_url]
        assert list(again.pulled) == [d] and [r.id for r in again.entries[d].runs("").runs] == ["r1"]
    finally:
        again.close()


def test_a_directory_moves_to_another_link_when_its_link_is_removed(make: Make) -> None:
    a, _, a_url = make("A")
    (b, _, b_url), (c, _, _) = make("B", crawls=False), make("C", crawls=False)
    b.add_link(a_url)
    c.add_link(a_url)
    c.add_link(b_url)
    assert wait_for(lambda: vias(c, a, b, c) == {own(a): ["A", "C"]} and synced(c))
    mirror = c.entries[own(a)]
    pull = mirror.origin
    assert isinstance(pull, Pull)
    c.remove(a_url)
    assert wait_for(lambda: vias(c, a, b, c) == {own(a): ["A", "B", "C"]})
    assert c.entries[own(a)] is mirror and c.pulled[own(a)].link == b_url
    assert wait_for(lambda: pull.connected) and [r.id for r in mirror.runs("").runs] == ["r1"]


def test_links_are_saved_and_pulled_again_after_a_restart(make: Make, tmp_path: Path) -> None:
    a, _, a_url = make("A")
    b, _, _ = make("B", crawls=False)
    b.add_link(a_url)
    b.close()
    again = Node(tmp_path / "B" / "cache", tmp_path / "B" / "state" / "roots.json")
    try:
        again.load()
        assert again.identity == b.identity and [link.url for link in again.links_info()] == [a_url]
        assert wait_for(lambda: list(again.pulled) == [own(a)] and synced(again))
    finally:
        again.close()


def test_links_are_added_listed_and_removed_over_http(make: Make) -> None:
    a, _, a_url = make("A")
    b, _, b_url = make("B", crawls=False)
    assert post_json(f"{b_url}/api/node/add", {"path": a_url + "/"}) == (200, {"name": None, "id": None, "url": "/"})
    info = get_json(f"{b_url}/api/node")
    assert info["node"] == {"id": b.identity.id, "name": "B"}
    assert info["links"] == [{"url": a_url, "id": a.identity.id, "name": "A", "state": "connected", "error": ""}]
    assert info["nodes"] == [{"id": a.identity.id, "name": "A", "via": [a.identity.id]}]
    assert [(r["name"], r["root"], r["link"], r["via"]) for r in info["dirs"]] == [("runs", own(a), a_url, [a.identity.id])]
    status, body = post_json(f"{b_url}/api/node/remove", {"name": "runs"})
    assert status == 400 and a_url in body["error"]
    assert post_json(f"{b_url}/api/node/remove", {"name": a_url}) == (200, {"ok": True})
    assert get_json(f"{b_url}/api/node")["dirs"] == [] and b.history() == [a_url]
    assert post_json(f"{b_url}/api/node/add", {"path": "http://127.0.0.1:1"})[0] == 400


def test_a_trex_tells_of_every_node_it_reaches_by_the_nodes_between(make: Make) -> None:
    (a, _, _), (b, _, b_url), (c, _, c_url) = make("A", crawls=False), make("B", crawls=False), make("C")
    b.add_link(c_url)
    a.add_link(b_url)
    named = {x.identity.id: x.identity.name for x in (a, b, c)}
    assert wait_for(lambda: {named[p.id]: [named[v] for v in p.via] for p in a.peers()} == {"B": ["B"], "C": ["C", "B"]})
    assert [(named[p.id], [named[v] for v in p.via]) for p in a.holdings().nodes or []] == [("A", ["A"]), ("B", ["B", "A"]), ("C", ["C", "B", "A"])]
    assert [(d.name, [named[v] for v in d.via]) for d in a.served()] == [("runs", ["C", "B"])]


def test_an_add_passed_along_links_is_tracked_by_the_last_node_and_pulled_by_the_others(make: Make, tmp_path: Path) -> None:
    (a, _, a_url), (b, _, b_url), (c, _, c_url) = make("A", crawls=False), make("B", crawls=False), make("C", crawls=False)
    b.add_link(c_url)
    a.add_link(b_url)
    assert wait_for(lambda: len(a.peers()) == 2)
    far = tmp_path / "files" / "far" / "runs"
    write_run(far / "r1")
    status, body = post_json(f"{a_url}/api/node/add", {"path": str(far), "at": [b.identity.id, c.identity.id]})
    d = f"C:{far}"
    assert (status, body["id"]) == (200, d) and c.tracked() == [str(far)] and b.tracked() == a.tracked() == []
    assert d in b.pulled and d in a.pulled and a.pulled[d].via == [c.identity.id, b.identity.id]
    near = tmp_path / "files" / "near" / "runs"
    write_run(near / "r1")
    assert a.add_at(str(near), [b.identity.id]) == f"B:{near}" and b.tracked() == [str(near)] and f"B:{near}" in a.pulled


def test_an_add_a_node_on_the_way_refuses_says_why(make: Make, tmp_path: Path) -> None:
    (a, _, _), (b, _, b_url), (c, _, c_url) = make("A", crawls=False), make("B", crawls=False), make("C", crawls=False)
    b.add_link(c_url)
    a.add_link(b_url)
    assert wait_for(lambda: len(a.peers()) == 2)
    with pytest.raises(ValueError, match="missing is not a directory"):
        a.add_at(str(tmp_path / "missing"), [b.identity.id, c.identity.id])
    with pytest.raises(ValueError, match="has no link to node nobody"):
        a.add_at(str(tmp_path), [b.identity.id, "nobody"])
    a.links[b_url].peers = None  # as from a trex that tells of no nodes
    with pytest.raises(ValueError, match="passes no add on"):
        a.add_at(str(tmp_path), [b.identity.id, c.identity.id])
    assert c.tracked() == b.tracked() == []


def test_a_trex_will_not_pull_from_itself(make: Make) -> None:
    a, _, a_url = make("A")
    with pytest.raises(ValueError, match="is this trex"):
        a.add_link(a_url)


def test_every_node_serves_its_directories_by_id_and_tells_what_it_holds(make: Make, tmp_path: Path,
                                                                         http_server: Callable[[Server], str]) -> None:
    a, _, a_url = make("A")
    assert wait_for(lambda: [r["id"] for r in get_json(f"{a_url}{dir_base(own(a))}/api/runs")["runs"]] == ["r1"])
    alone = Explorer(Crawl(tmp_path / "files" / "A" / "runs"), tmp_path / "alone-cache")
    alone.sync()
    temporary = node_of(alone)
    url = http_server(server.serve(temporary, "127.0.0.1", 0))
    holdings = get_json(f"{url}/api/holdings")
    me = temporary.identity
    assert holdings == temporary.holdings().wire() == {"node": me.wire(), "dirs": [{"id": temporary.home, "via": [me.id]}],
                                                       "nodes": [{"id": me.id, "name": me.name, "via": [me.id]}]}
    assert [r["id"] for r in get_json(f"{url}{dir_base(str(temporary.home))}/api/runs")["runs"]] == ["r1"]
    assert request(f"{url}/d/{urllib.parse.quote('elsewhere:/x', safe='')}/api/runs")[0] == 404
