"""Tests – laufen mit `python test_simulate.py` oder `pytest`."""

import tomllib
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import re19watch
from simulate import FAST, REGIONAL, blocks, make_train, resolve

TZ = ZoneInfo("Europe/Berlin")
CFG = tomllib.loads((Path(__file__).parent / "config.toml").read_text(encoding="utf-8"))
C = re19watch.build_corridor(CFG["corridor"])
DAY = datetime(2026, 9, 24, tzinfo=TZ)


def at(h, m, s=0):
    return DAY.replace(hour=h, minute=m, second=s)


def my_re19(dep=at(7, 5)):
    return make_train(C, "RE 19", 2, REGIONAL, "Rödental", "Coburg", "Dörfles-Esbach", dep)


def wait_at_mine(me, others):
    r = resolve(me, others, C)
    return r.times["Dörfles-Esbach"] - me.times["Dörfles-Esbach"], r


def test_blocks():
    assert blocks(["Rödental", "Abzweig SFS", "Dörfles-Esbach", "Coburg Nord", "Coburg"], {"Rödental", "Coburg"}) == [
        ["Rödental", "Abzweig SFS", "Dörfles-Esbach", "Coburg Nord", "Coburg"]
    ]
    assert blocks(["A", "B", "C"], {"B"}) == [["A", "B"], ["B", "C"]]


def test_no_conflict_when_ice_on_time():
    ice = make_train(C, "ICE → Berlin", 4, FAST, "Coburg", "Abzweig SFS", "Coburg", at(6, 55))
    wait, _ = wait_at_mine(my_re19(), [ice])
    assert wait == timedelta(0)


def test_delayed_opposing_ice_holds_re19():
    # ICE Richtung Berlin 7 min zu spät -> belegt eingleisig Coburg..Abzweig bis 07:06
    ice = make_train(C, "ICE → Berlin", 4, FAST, "Coburg", "Abzweig SFS", "Coburg", at(7, 2))
    wait, r = wait_at_mine(my_re19(), [ice])
    assert wait == timedelta(minutes=2, seconds=30), wait  # ICE frei 07:05:30 + 1 min Puffer, RE wäre 07:04 am Abzweig
    assert r.holds and r.holds[0].at == "Rödental" and r.holds[0].conflicts[0].kind == "gegenzug"


def test_merging_ice_same_direction_goes_first():
    # ICE aus Berlin schleift am Abzweig ein, gleiche Richtung, knapp hinter dem RE
    ice = make_train(C, "ICE aus Berlin", 4, FAST, "Abzweig SFS", "Coburg", "Coburg", at(7, 10))
    wait, r = wait_at_mine(my_re19(), [ice])
    assert wait == timedelta(minutes=4, seconds=30), wait
    assert r.holds[0].conflicts[0].kind == "vorrang"


def test_lower_priority_later_train_does_not_hold_me():
    rb = make_train(C, "RB → Sonneberg", 1, REGIONAL, "Coburg", "Rödental", "Coburg", at(7, 3))
    wait, _ = wait_at_mine(my_re19(), [rb])
    assert wait == timedelta(0)


def _ev(line, product):
    return re19watch.Event("departures", None, line, product, None, None, None, None, False, "")


def test_line_priority_override():
    assert re19watch.priority(_ev("RE29", "regional"), CFG) == 3
    assert re19watch.priority(_ev("RE 19", "regional"), CFG) == 2
    assert re19watch.priority(_ev("ICE 501", "nationalExpress"), CFG) == 4


def test_re29_to_erfurt_holds_late_re19():
    # RE 19 5 min zu spät (Dörfles-Esbach 07:25), RE 29 → Erfurt pünktlich Coburg ab 07:27: RE 29 hat Vorrang
    me = make_train(C, "RE 19", 2, REGIONAL, "Rödental", "Coburg", "Dörfles-Esbach", at(7, 25))
    re29 = make_train(C, "RE 29 → Erfurt", 3, REGIONAL, "Coburg", "Abzweig SFS", "Coburg", at(7, 27))
    wait, r = wait_at_mine(me, [re29])
    assert wait > timedelta(0), wait
    assert r.holds[0].conflicts[0].kind == "gegenzug"
    # gleiche Lage mit RE-Priorität 2: Gleichstand, wer zuerst kommt -> RE 19 wartet nicht
    re29_low = make_train(C, "RE 29 → Erfurt", 2, REGIONAL, "Coburg", "Abzweig SFS", "Coburg", at(7, 27))
    assert wait_at_mine(me, [re29_low])[0] < wait


def test_parse_timetables_xml():
    import xml.etree.ElementTree as ET

    plan = ET.fromstring(
        '<s id="-67082-2609240418-12"><tl c="ICE" n="501"/>'
        '<ar pt="2609240703" ppth="Leipzig Hbf|Erfurt Hbf"/><dp pt="2609240705" ppth="Bamberg|München Hbf"/></s>'
    )
    chg = ET.fromstring('<s id="-67082-2609240418-12"><ar ct="2609240711"/><dp ct="2609240713" cs="c"/></s>')
    ar = re19watch.parse_tt_stop(plan, chg, "arrivals", TZ)
    assert (ar.line, ar.product, ar.other_end) == ("ICE 501", "nationalExpress", "Leipzig Hbf")
    assert ar.planned == at(7, 3) and ar.when == at(7, 11) and ar.delay_min == 8 and not ar.cancelled
    assert ar.trip_id == "-67082-2609240418"
    dp = re19watch.parse_tt_stop(plan, chg, "departures", TZ)
    assert dp.cancelled and dp.other_end == "München Hbf"
    assert re19watch.parse_tt_stop(plan, None, "departures", TZ).when == at(7, 5)


def test_end_to_end_with_fake_api():
    """Kompletter Durchlauf run_check() mit gefälschten API-Antworten."""

    def ev(line, product, planned, delay_min, other, key, trip):
        p = planned
        return {
            "tripId": trip,
            "line": {"name": line, "product": product, "fahrtNr": trip},
            "plannedWhen": p.isoformat(),
            "when": (p + timedelta(minutes=delay_min)).isoformat(),
            "delay": delay_min * 60,
            key: other,
        }

    class FakeClient:
        def board(self, sid, kind, when, duration=60):
            if sid == CFG["stations"]["mine_id"]:
                raw = [ev("RE 19", "regionalExpress", at(7, 20), 0, "Nürnberg Hbf", "direction", "re19")]
            elif kind == "departures":
                raw = [
                    ev("ICE 1004", "nationalExpress", at(7, 10), 7, "Berlin Hbf", "direction", "ice"),
                    ev("RE 14", "regionalExpress", at(7, 13), 0, "Lichtenfels", "direction", "re14"),
                ]
            else:
                raw = [ev("RE 19", "regionalExpress", at(7, 25), 0, "Sonneberg(Thür)Hbf", "provenance", "re19")]
            return [re19watch.parse_event(e, kind) for e in raw]

    text, info = re19watch.run_check(CFG, at(6, 55), client=FakeClient())
    print(text)
    assert info["est"].extra_vs_db == timedelta(minutes=2, seconds=30)
    assert "ICE 1004" in text and "⚠️" in text


def test_mock_full_pipeline():
    """--mock --dry-run: Parser, Simulation und Nachricht komplett, ohne Internet."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert re19watch.main(["--mock", "--dry-run"]) == 0
    out = buf.getvalue()
    assert "ICE 1606" in out and "⚠️" in out and "fehlgeschlagen" not in out, out
    assert "🚨 Alarm: RE 19 verspätet | +15 min – Abfahrt ~07:35" in out and "Loslaufen um 07:30" in out, out


def test_collect_rows_and_prediction_record():
    import xml.etree.ElementTree as ET

    import collect

    chg = ET.fromstring(
        '<timetable><s id="-67082-2609240418-12"><m t="h" cat="Bauarbeiten"/>'
        '<ar ct="2609240711"><m t="d" c="43" ts="2609240650"/></ar><dp ct="2609240713" cs="c"/></s></timetable>'
    )
    rows = collect.change_rows(chg, "8001338", at(6, 55), TZ)
    assert [r["kind"] for r in rows] == ["arrival", "departure"]
    assert rows[0]["changed"] == at(7, 11).isoformat() and rows[0]["messages"][0]["c"] == "43"
    assert rows[0]["stop_messages"][0]["cat"] == "Bauarbeiten" and rows[1]["status"] == "c"
    assert rows[0]["trip_id"] == "-67082-2609240418"

    now = at(6, 55)
    _, info = re19watch.run_check(CFG, now, client=re19watch.MockTimetablesClient(TZ, now))
    rec = collect.prediction_record(now, info, "test")
    assert rec["expected_delay_min"] == 15 and rec["holds"][0]["conflicts"][0]["with"].startswith("ICE 1606")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"✔ {name}")
    print("Alle Tests grün.")
