#!/usr/bin/env python3
"""
re19watch.py – Morgen-Check: Wird mein RE 19 ab Doerfles-Esbach durch einen
Zug mit Vorrang (ICE/Gegenzug) auf dem eingleisigen Abschnitt aufgehalten?

Ablauf:
  1. Mein Zug aus der Abfahrtstafel Doerfles-Esbach holen (Soll + DB-Prognose).
  2. Alle Zuege, die in Coburg in/aus dem Korridor fahren, aus den Tafeln
     Coburg Ab/An holen und nach Laufweg einordnen (Werrabahn oder Einschleifung
     zur Schnellfahrstrecke).
  3. Belegung simulieren (simulate.py), Ergebnis mit DB-Prognose vergleichen.
  4. Telegram-Nachricht (oder Konsolenausgabe mit --dry-run).

Datenquelle: DB API Marketplace "Timetables" (Env DB_CLIENT_ID, DB_API_KEY),
alternativ selbst gehosteter db-vendo-client ([api] provider = "vendo").
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from simulate import FAST, REGIONAL, Corridor, Estimate, Train, estimate, make_train

HERE = Path(__file__).resolve().parent

PRIORITY = {"nationalExpress": 4, "national": 3, "regionalExpress": 2, "regional": 1, "suburban": 1}


def priority(ev: "Event", cfg: dict, default: int = 0) -> int:
    """Vorrang nach Produkt; einzelne Linien per [priority.lines] in config.toml überschreibbar."""
    overrides = {norm_line(k): v for k, v in cfg.get("priority", {}).get("lines", {}).items()}
    return overrides.get(norm_line(ev.line), PRIORITY.get(ev.product, default))


# --------------------------------------------------------------------------- Daten


@dataclass
class Event:
    kind: str  # "departures" | "arrivals"
    trip_id: str | None
    line: str
    product: str
    fahrt_nr: str | None
    planned: datetime | None
    when: datetime | None
    delay_min: float | None
    cancelled: bool
    other_end: str  # Ziel (Abfahrt) bzw. Herkunft (Ankunft)


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def parse_event(e: dict, kind: str) -> Event:
    line = e.get("line") or {}
    delay = e.get("delay")
    return Event(
        kind=kind,
        trip_id=e.get("tripId"),
        line=line.get("name") or "?",
        product=line.get("product") or "regional",
        fahrt_nr=str(line.get("fahrtNr")) if line.get("fahrtNr") else None,
        planned=_dt(e.get("plannedWhen")),
        when=_dt(e.get("when")),
        delay_min=delay / 60 if delay is not None else None,
        cancelled=bool(e.get("cancelled")),
        other_end=(e.get("direction") if kind == "departures" else e.get("provenance")) or "",
    )


class DbClient:
    def __init__(self, base_url: str, timeout: int = 20):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, **params):
        r = requests.get(self.base + path, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def station_id(self, name: str) -> str:
        res = self._get("/locations", query=name, results=5, addresses="false", poi="false")
        for loc in res:
            if loc.get("type") in ("stop", "station"):
                return loc["id"]
        raise RuntimeError(f"Station '{name}' nicht gefunden")

    def board(self, station_id: str, kind: str, when: datetime, duration: int = 60) -> list[Event]:
        data = self._get(f"/stops/{station_id}/{kind}", when=when.isoformat(), duration=duration, results=100)
        items = data.get(kind, []) if isinstance(data, dict) else data
        return [parse_event(e, kind) for e in items]


# DB API Marketplace "Timetables" (IRIS): Soll-Plan je Bahnhof+Stunde, Echtzeit über /fchg.
TT_BASE = "https://apis.deutschebahn.com/db-api-marketplace/apis/timetables/v1"
TT_PRODUCT = {"ICE": "nationalExpress", "IC": "national", "EC": "national", "RE": "regionalExpress",
              "IRE": "regionalExpress", "S": "suburban"}

ML_FEATURES = [
    "db_delay_now", "up_sonneberg", "up_neustadt", "up_moenchroeden", "up_roedentalmitte", "up_roedental",
    "n_upstream_known", "last_known_delay", "max_upstream_delay", "upstream_trend",
    "conflict_max_delay", "conflict_mean_delay", "n_conflict_trips", "ice_max_delay", "n_ice_nearby",
    "conflict_gap_min", "ice_gap_min", "vorgaenger_delay", "vorgaenger_luecke_min",
    "hour", "minute_of_day", "dow", "month", "is_weekend",
]
VOR_WINDOW_MIN = 45   # muss zu VOR_WINDOW in ml/build_features.py passen

# Wie weit die echte Verspaetung typischerweise UEBER der Prognose liegt (p90 des
# Residuums), gemessen ueber fuenf gleitende Zweimonatsfenster 2025 (n=6465):
#   Lage                            n     p90    Anteil Ist >= 5 min
#   alle                         6465   +2,9                  18,7 %
#   ICE nah (<=15 min) und >=10   334   +4,1                  24,9 %
#   vorausfahrender Zug >= 8 min  357   +7,5                  65,3 %
# Daraus die Obergrenze der angezeigten Spanne. Der Punktwert bleibt die Prognose -
# so wird nicht grundlos Alarm geschlagen, die Unsicherheit aber benannt.
RISIKO_BASIS = 3.0
RISIKO_ICE = 4.0
RISIKO_VORGAENGER = 8.0
UPSTREAM = [("8013008", "sonneberg"), ("8004325", "neustadt"), ("8004064", "moenchroeden"),
            ("8005122", "roedentalmitte"), ("8004633", "roedental")]


def _tt_time(s: str | None, tz: ZoneInfo) -> datetime | None:
    return datetime.strptime(s, "%y%m%d%H%M").replace(tzinfo=tz) if s else None


def parse_tt_stop(s: ET.Element, chg: ET.Element | None, kind: str, tz: ZoneInfo) -> Event | None:
    """Ein <s>-Eintrag aus /plan plus passender Eintrag aus /fchg (gleiche id) -> Event."""
    tag = "dp" if kind == "departures" else "ar"
    ev = s.find(tag)
    if ev is None:
        return None
    ch = chg.find(tag) if chg is not None else None
    tl = s.find("tl")
    tla = tl.attrib if tl is not None else {}
    cat, nr = tla.get("c", ""), tla.get("n")
    planned = _tt_time(ev.get("pt"), tz)
    when = _tt_time(ch.get("ct"), tz) if ch is not None and ch.get("ct") else planned
    path = ((ch.get("cpth") if ch is not None else None) or ev.get("ppth") or "").split("|")
    status = (ch.get("cs") if ch is not None else None) or ev.get("cs")
    return Event(
        kind=kind,
        trip_id=s.get("id", "").rsplit("-", 1)[0] or None,  # id = <fahrt>-<start>-<halt-index>
        line=ev.get("l") or f"{cat} {nr}".strip(),
        product=TT_PRODUCT.get(cat, "regional"),
        fahrt_nr=nr,
        planned=planned,
        when=when,
        delay_min=(when - planned).total_seconds() / 60 if when and planned else None,
        cancelled=status == "c",
        other_end=(path[-1] if kind == "departures" else path[0]) if path else "",
    )


class TimetablesClient:
    def __init__(self, client_id: str, api_key: str, tz: ZoneInfo, timeout: int = 20):
        self.headers = {"DB-Client-Id": client_id, "DB-Api-Key": api_key, "accept": "application/xml"}
        self.tz = tz
        self.timeout = timeout
        self._cache: dict[str, ET.Element] = {}

    def _get(self, path: str) -> ET.Element:
        if path not in self._cache:
            r = requests.get(f"{TT_BASE}/{path}", headers=self.headers, timeout=self.timeout)
            if r.status_code == 404:  # keine Fahrten in dieser Stunde
                self._cache[path] = ET.Element("timetable")
            else:
                r.raise_for_status()
                self._cache[path] = ET.fromstring(r.content)
        return self._cache[path]

    def station_id(self, name: str) -> str:
        for st in self._get(f"station/{requests.utils.quote(name)}"):
            if st.get("name", "").lower() == name.lower():
                return st.get("eva")
        raise RuntimeError(f"Station '{name}' nicht gefunden – EVA-Nummer in config.toml eintragen")

    def board(self, station_id: str, kind: str, when: datetime, duration: int = 60) -> list[Event]:
        when = when.astimezone(self.tz)
        end = when + timedelta(minutes=duration)
        changes = {s.get("id"): s for s in self._get(f"fchg/{station_id}")}
        search_start = when - timedelta(minutes=120)
        out, seen, h = [], set(), search_start.replace(minute=0, second=0, microsecond=0)
        while h < end:
            for s in self._get(f"plan/{station_id}/{h:%y%m%d}/{h:%H}"):
                sid = s.get("id")
                ev = parse_tt_stop(s, changes.get(sid), kind, self.tz)
                in_window = ev and ev.planned and when <= ev.planned < end
                delayed_into_window = ev and ev.when and when <= ev.when < end
                if ev and sid not in seen and (in_window or delayed_into_window):
                    seen.add(sid)
                    out.append(ev)
            h += timedelta(hours=1)
        return sorted(out, key=lambda e: e.planned)


class MockTimetablesClient(TimetablesClient):
    """Erfundene Timetables-Antworten für heute (echtes XML-Format, echter Parser).
    Szenario: ICE → Berlin 8 min zu spät, belegt den eingleisigen Abschnitt, während der RE 19 kommt."""

    def __init__(self, tz: ZoneInfo, day: datetime):
        super().__init__("mock", "mock", tz)
        d = f"{day:%y%m%d}"
        self._plans = {
            "8001484": [
                f'<s id="4905-{d}0703-6"><tl c="RE" n="4905"/>'
                f'<ar pt="{d}0720" l="RE19" ppth="Sonneberg(Thür)Hbf|Rödental"/>'
                f'<dp pt="{d}0720" l="RE19" ppth="Coburg Nord|Coburg|Bamberg|Nürnberg Hbf"/></s>',
            ],
            "8001338": [
                f'<s id="4905-{d}0703-8"><tl c="RE" n="4905"/>'
                f'<ar pt="{d}0726" l="RE19" ppth="Sonneberg(Thür)Hbf|Dörfles-Esbach"/>'
                f'<dp pt="{d}0737" l="RE19" ppth="Bamberg|Nürnberg Hbf"/></s>',
                f'<s id="1606-{d}0540-4"><tl c="ICE" n="1606"/>'
                f'<ar pt="{d}0710" ppth="München Hbf|Nürnberg Hbf|Bamberg"/>'
                f'<dp pt="{d}0712" ppth="Erfurt Hbf|Halle(Saale)Hbf|Berlin Hbf"/></s>',
                f'<s id="4900-{d}0611-7"><tl c="RE" n="4900"/>'
                f'<ar pt="{d}0720" l="RE29" ppth="Nürnberg Hbf|Bamberg"/>'
                f'<dp pt="{d}0727" l="RE29" ppth="Dörfles-Esbach|Erfurt Hbf"/></s>',
                f'<s id="501-{d}0418-12"><tl c="ICE" n="501"/>'
                f'<ar pt="{d}0703" ppth="Leipzig Hbf|Erfurt Hbf"/>'
                f'<dp pt="{d}0705" ppth="Bamberg|München Hbf"/></s>',
            ],
        }
        self._changes = {"8001338": [f'<s id="1606-{d}0540-4"><ar ct="{d}0718"/><dp ct="{d}0720"/></s>']}

    def _get(self, path: str) -> ET.Element:
        kind, eva = path.split("/")[:2]
        items = (self._plans if kind == "plan" else self._changes).get(eva, [])
        return ET.fromstring(f"<timetable>{''.join(items)}</timetable>")


def make_client(api: dict, tz: ZoneInfo):
    if api.get("provider", "timetables") == "timetables":
        cid, key = os.environ.get("DB_CLIENT_ID"), os.environ.get("DB_API_KEY")
        if not cid or not key:
            raise RuntimeError("DB_CLIENT_ID/DB_API_KEY fehlen (DB API Marketplace, Timetables)")
        return TimetablesClient(cid, key, tz)
    return DbClient(api["base_url"])


# --------------------------------------------------------------------------- Hilfen


def norm_line(s: str) -> str:
    return "".join((s or "").split()).upper()


def train_class(product: str) -> str:
    return FAST if product in ("nationalExpress", "national") else REGIONAL


def same_trip(a: Event, b: Event) -> bool:
    if a.trip_id and b.trip_id and a.trip_id == b.trip_id:
        return True
    return bool(a.fahrt_nr) and a.fahrt_nr == b.fahrt_nr and norm_line(a.line) == norm_line(b.line)


def classify(other_end: str, cls_cfg: dict) -> str | None:
    """Laeuft der Zug ueber die Einschleifung zur SFS oder ueber die Werrabahn Richtung Roedental?"""
    txt = other_end.lower()
    if any(k.lower() in txt for k in cls_cfg["via_sfs"]):
        return "sfs"
    if any(k.lower() in txt for k in cls_cfg["via_werrabahn"]):
        return "werrabahn"
    return None


def hhmm(dt: datetime, tz: ZoneInfo) -> str:
    return dt.astimezone(tz).strftime("%H:%M")


def minutes(td: timedelta) -> int:
    return math.floor(td.total_seconds() / 60 + 0.5)  # halbe Minuten aufrunden


def delay_suffix(ev: Event) -> str:
    return f" (+{round(ev.delay_min)} min)" if ev.delay_min and ev.delay_min >= 1 else ""


def build_corridor(cc: dict) -> Corridor:
    run, double = {}, set()
    for s in cc["segment"]:
        pts = cc["points"]
        a, b = (s["from"], s["to"]) if pts.index(s["from"]) < pts.index(s["to"]) else (s["to"], s["from"])
        run[(a, b)] = {REGIONAL: float(s["regional"]), FAST: float(s["fast"])}
        if s.get("double_track"):
            double.add((a, b))
    return Corridor(
        points=cc["points"],
        run_min=run,
        crossing_points=set(cc["crossing_points"]),
        double_track=double,
        opposing_buffer_min=float(cc.get("opposing_buffer_min", 1.0)),
        following_headway_min=float(cc.get("following_headway_min", 2.0)),
    )


def build_others(deps: list[Event], arrs: list[Event], cfg: dict, c: Corridor, use_prog: bool, tz) -> list[Train]:
    cc = cfg["corridor"]
    coburg, junction, wb_end = cc["coburg_point"], cc["junction_point"], cc["werrabahn_end"]
    out: list[Train] = []
    for ev in deps + arrs:
        if ev.cancelled or ev.planned is None:
            continue
        route = classify(ev.other_end, cfg["classify"])
        if route is None:
            continue
        far_end = junction if route == "sfs" else wb_end
        t = ev.when if (use_prog and ev.when) else ev.planned
        if ev.kind == "departures":  # verlaesst Coburg Richtung Norden
            label = f"{ev.line} nach {ev.other_end}{delay_suffix(ev)}"
            start, end = coburg, far_end
        else:  # kommt aus Norden nach Coburg
            label = f"{ev.line} aus {ev.other_end}{delay_suffix(ev)}"
            start, end = far_end, coburg
        out.append(
            make_train(c, label, priority(ev, cfg), train_class(ev.product), start, end, coburg, t, ev.trip_id)
        )
    return out


def _event_delay_at(client: DbClient, trip_id: str | None, station_id: str, planned: datetime) -> float | None:
    if not trip_id:
        return None
    for kind in ("departures", "arrivals"):
        for ev in client.board(station_id, kind, planned - timedelta(minutes=60), 120):
            if ev.trip_id == trip_id and ev.delay_min is not None:
                return ev.delay_min
    return None


def ml_prediction(client: DbClient, cfg: dict, me_ev: Event, deps: list[Event], arrs: list[Event],
                  planned: datetime, now: datetime,
                  my_board: list[Event] | None = None) -> tuple[datetime | None, float | None]:
    """Point-in-time ML-Schaetzung; bei fehlendem Artefakt bleibt die Regelpipeline aktiv."""
    try:
        import joblib
        import numpy as np
        import pandas as pd

        model = joblib.load(HERE / "ml" / "model_werrabahn.joblib")
    except (ImportError, OSError, ValueError):
        return None, None, {"zuschlag": RISIKO_BASIS, "grund": ""}

    row = {feature: np.nan for feature in ML_FEATURES}
    row["db_delay_now"] = me_ev.delay_min if me_ev.delay_min is not None else np.nan
    upstream = []
    station_ids = {str(s["eva"]): str(s["eva"]) for s in cfg.get("collect", {}).get("stations", [])}
    for eva, name in UPSTREAM:
        delay = _event_delay_at(client, me_ev.trip_id, station_ids.get(eva, eva), planned)
        row["up_" + name] = delay if delay is not None else np.nan
        upstream.append(row["up_" + name])
    known = [value for value in upstream if not np.isnan(value)]
    row["n_upstream_known"] = len(known)
    row["last_known_delay"] = known[-1] if known else np.nan
    row["max_upstream_delay"] = max(known) if known else np.nan
    row["upstream_trend"] = known[-1] - known[0] if known else np.nan

    # WICHTIG: exakt dieselbe Zugauswahl wie im Training (ml/build_features.py),
    # sonst sieht das Modell im Betrieb eine andere Feature-Verteilung als beim
    # Lernen. Trainiert wurde nur auf RE28 (Gegenrichtung) sowie RE29 und ICE
    # (beide ueber die SFS-Einschleifung) - nicht auf RB24, RE32 oder S-Bahnen.
    def _ist_konfliktzug(ev: Event) -> bool:
        return norm_line(ev.line) in ("RE28", "RE29") or ev.product == "nationalExpress"

    conflicts = [ev for ev in deps + arrs
                 if _ist_konfliktzug(ev) and (ev.when or ev.planned) and
                 abs((ev.when or ev.planned) - planned) <= timedelta(minutes=30)]
    delays = [ev.delay_min for ev in conflicts if ev.delay_min is not None]
    ice = [ev for ev in conflicts if ev.product == "nationalExpress"]
    row["conflict_max_delay"] = max(delays) if delays else np.nan
    row["conflict_mean_delay"] = float(np.mean(delays)) if delays else np.nan
    row["n_conflict_trips"] = len({ev.trip_id for ev in conflicts})
    row["ice_max_delay"] = max((ev.delay_min for ev in ice if ev.delay_min is not None), default=np.nan)
    row["n_ice_nearby"] = len({ev.trip_id for ev in ice})
    passages = [ev.when for ev in conflicts if ev.when]
    ice_passages = [ev.when for ev in ice if ev.when]
    row["conflict_gap_min"] = min((abs((value - planned).total_seconds()) / 60 for value in passages), default=np.nan)
    row["ice_gap_min"] = min((abs((value - planned).total_seconds()) / 60 for value in ice_passages), default=np.nan)
    # Vorausfahrender Zug auf dem eingleisigen Abschnitt (staerkstes Feature des
    # Modells). Gleiche Auswahl wie im Training: letzte RE19/RE28/RE29-Abfahrt in
    # Doerfles-Esbach vor meiner, maximal VOR_WINDOW_MIN Minuten zurueck.
    # Die eigene Abfahrtstafel beginnt erst 10 min vor meiner Abfahrt, deshalb wird
    # das frueher liegende Fenster hier zusaetzlich geholt.
    try:
        mine_id = cfg["stations"].get("mine_id") or client.station_id(cfg["stations"]["mine"])
        board = list(my_board or [])
        board += client.board(mine_id, "departures",
                              planned - timedelta(minutes=VOR_WINDOW_MIN), VOR_WINDOW_MIN)
    except Exception:
        board = list(my_board or [])
    vor = [ev for ev in board
           if norm_line(ev.line) in ("RE19", "RE28", "RE29") and ev.planned is not None
           and not same_trip(ev, me_ev)
           and planned - timedelta(minutes=VOR_WINDOW_MIN) <= ev.planned < planned]
    if vor:
        letzter = max(vor, key=lambda ev: ev.planned)
        if letzter.delay_min is not None:
            row["vorgaenger_delay"] = letzter.delay_min
        row["vorgaenger_luecke_min"] = (planned - letzter.planned).total_seconds() / 60

    local = planned.astimezone(now.tzinfo)
    row.update(hour=local.hour, minute_of_day=local.hour * 60 + local.minute, dow=local.weekday(),
               month=local.month, is_weekend=int(local.weekday() >= 5))
    prediction = float(model.predict(pd.DataFrame([row])[ML_FEATURES])[0])

    # Lageabhaengige Obergrenze bestimmen (gemessene p90-Streuung, siehe oben).
    zuschlag, grund = RISIKO_BASIS, ""
    vd, ig, imd = row["vorgaenger_delay"], row["ice_gap_min"], row["ice_max_delay"]
    if not np.isnan(ig) and not np.isnan(imd) and ig <= 15 and imd >= 10:
        zuschlag, grund = RISIKO_ICE, "ein ICE fährt verspätet dicht vor dir über den Abschnitt"
    if not np.isnan(vd) and vd >= 8:
        zuschlag, grund = RISIKO_VORGAENGER, f"der Zug vor dir ist {vd:.0f} min zu spät"
    return (planned + timedelta(minutes=max(0.0, prediction)), prediction,
            {"zuschlag": zuschlag, "grund": grund})


def find_my_train(events: list[Event], mt: dict, tz: ZoneInfo) -> Event | None:
    want_line = norm_line(mt["line"])
    hh, mm = map(int, mt["planned_departure"].split(":"))
    for ev in events:
        if norm_line(ev.line) != want_line or ev.planned is None:
            continue
        p = ev.planned.astimezone(tz)
        if (p.hour, p.minute) != (hh, mm):
            continue
        if mt.get("direction_contains") and mt["direction_contains"].lower() not in ev.other_end.lower():
            continue
        return ev
    return None


# --------------------------------------------------------------------------- Kern


def run_check(cfg: dict, now: datetime, client: DbClient | None = None) -> tuple[str, dict]:
    tz = now.tzinfo
    mt, st, api = cfg["my_train"], cfg["stations"], cfg["api"]
    client = client or make_client(api, tz)
    c = build_corridor(cfg["corridor"])

    hh, mm = map(int, mt["planned_departure"].split(":"))
    planned_dt = datetime.combine(now.date(), time(hh, mm), tzinfo=tz)
    head = f"🚆 <b>{html.escape(mt['line'])} · {html.escape(mt['point'])} {mt['planned_departure']}</b>"

    mine_id = st.get("mine_id") or client.station_id(st["mine"])
    my_board = client.board(mine_id, "departures", planned_dt - timedelta(minutes=10), 60)
    me_ev = find_my_train(my_board, mt, tz)
    if me_ev is None:
        return f"{head}\n❓ Heute nicht in der Abfahrtstafel gefunden (Feiertag, Baustelle, Fahrplanwechsel?).", {}
    if me_ev.cancelled:
        return f"{head}\n❌ Laut DB fällt der Zug heute aus.", {}

    cob_id = st.get("coburg_id") or client.station_id(st["coburg"])
    # WICHTIG: Die Tafel wird nach SOLL-Zeit gefiltert, nicht nach tatsaechlicher
    # Durchfahrt. Ein stark verspaeteter Zug, dessen Soll-Zeit vor dem Fensterbeginn
    # liegt, war damit unsichtbar - obwohl er real genau in meinen Weg faehrt.
    # Gemessen an allen drei 07:20-Konfliktfaellen 2025: es war jedes Mal ICE 1604
    # mit Soll-Abfahrt 06:43 und +36..+40 min, also real 07:19-07:23. Bei den alten
    # 35 min Vorlauf begann die Tafel um 06:45 und hat ihn nie erfasst.
    vor = int(api.get("window_before_min", 90))
    since = planned_dt - timedelta(minutes=vor)
    dauer = vor + int(api.get("window_after_min", 30))
    deps = [e for e in client.board(cob_id, "departures", since, dauer) if not same_trip(e, me_ev)]
    arrs = [e for e in client.board(cob_id, "arrivals", since, dauer) if not same_trip(e, me_ev)]

    def me_train(use_prog: bool) -> Train:
        t = me_ev.when if (use_prog and me_ev.when) else me_ev.planned
        return make_train(
            c, f"{me_ev.line}", priority(me_ev, cfg, 2), train_class(me_ev.product),
            mt["from_point"], mt["to_point"], mt["point"], t, me_ev.trip_id,
        )

    db_when = me_ev.when or me_ev.planned
    est = estimate(
        me_train(False), build_others(deps, arrs, cfg, c, False, tz),
        me_train(True), build_others(deps, arrs, cfg, c, True, tz),
        c, mt["point"], me_ev.planned, db_when,
    )
    est.ml_estimate, _, risiko = ml_prediction(client, cfg, me_ev, deps, arrs, me_ev.planned,
                                               now, my_board=my_board)
    est.risiko_zuschlag, est.risiko_grund = risiko["zuschlag"], risiko["grund"]
    return format_message(head, est, me_ev, tz, int(mt.get("walk_min", 5))), {"est": est, "me": me_ev}


def expected_departure(est: Estimate) -> tuple[datetime, int]:
    """Erwartete Abfahrt am Einstieg und Verspätung in ganzen Minuten.

    Das ML-Modell wird DIREKT verwendet, nicht über max() mit der DB verrechnet.
    Gemessen an 127 eigenen Fahrten und 1616 Fällen aus 2025 ist das die beste
    Variante (MAE 1,78 gegenüber 1,80 für max(DB,ML) und 1,92 für die DB allein):
      - Das Modell sagt nur in 6 von 132 Fällen weniger als die DB voraus, liegt
        dort aber RICHTIGER (1,61 vs. 2,00 min). Die max()-Schranke verwarf also
        korrekte Abwärtskorrekturen.
      - Geprüfte Alternativen, die alle SCHLECHTER waren: ML nur bei erkanntem
        Konflikt (2,18-2,27), Mischungen mit der DB, pauschaler Sockelabzug.
        Sie gewinnen allenfalls auf der kleinen 2026er-Stichprobe und verlieren
        auf dem zehnmal größeren 2025er-Testsatz.

    Fällt das Modell aus oder liefert es einen unplausiblen Wert, greift die
    bisherige Regel-/DB-Logik als Rückfallebene.
    """
    dep = max(est.db_when, est.estimate)
    if est.ml_estimate is not None:
        vorsprung = (est.ml_estimate - est.planned).total_seconds() / 60
        if -5 <= vorsprung <= 180:        # Plausibilitätsgrenze gegen Ausreißer
            dep = est.ml_estimate

    # KEIN Veto des Regelmodells mehr. Begruendung aus dem Replay ueber das ganze
    # Jahr 2025 (ml/replay_2025.py, 4291 Zielzuege):
    #   - Das Regelmodell erkennt eine Blockade bei 13,7 % der Zuege, davon sind
    #     73 % Fehlalarme (Zug faehrt tatsaechlich mit unter 5 min Verspaetung).
    #   - Mit Veto wird der 07:20-Slot auf 4,19 min MAE verschlechtert, ohne Veto
    #     sind es 2,47 - schlechter also sogar als die DB-Prognose (3,03).
    #   - Auch auf die praezise Teilmenge begrenzt (rule_extra >= 8 min, dort 63 %
    #     Treffer) bleibt es schaedlich: 15,30 statt 5,37 min MAE.
    #   - rule_extra_min als zusaetzliches ML-Feature bringt ebenfalls nichts
    #     (Rang 25 von 25, Permutation Importance -0,003).
    # Das Regelmodell bleibt im Einsatz, aber nur noch fuer die Begruendung
    # ("welcher Zug kreuzt wo") und als Rueckfallebene, wenn das ML-Modell fehlt.

    delay = max(0, minutes(dep - est.planned))
    return est.planned + timedelta(minutes=delay), delay


def format_message(head: str, est: Estimate, me_ev: Event, tz: ZoneInfo, walk_min: int = 5) -> str:
    extra = minutes(est.extra_vs_db)
    dep, delay = expected_departure(est)
    leave = dep - timedelta(minutes=walk_min)

    db_delay = max(0, minutes(est.db_when - est.planned))
    lines = [head, ""]
    # Obergrenze = Prognose + gemessene p90-Streuung der aktuellen Lage. Nur zeigen,
    # wenn die Lage riskanter ist als normal (sonst steht in jeder Nachricht eine
    # Spanne und sie verliert ihre Aussage).
    oben = delay + int(round(est.risiko_zuschlag))
    spanne = est.risiko_zuschlag > RISIKO_BASIS and oben > delay + 1
    if delay <= 0 and not spanne:
        lines.append("✅ <b>Pünktlich</b>")
    elif spanne:
        lines.append(f"⚠️ <b>Ca. +{delay} min – Abfahrt ~{hhmm(dep, tz)}</b>")
        lines.append(f"<b>Kann bis +{oben} min werden</b> "
                     f"(~{hhmm(est.planned + timedelta(minutes=oben), tz)})")
        if est.risiko_grund:
            lines.append(f"<i>{html.escape(est.risiko_grund)}.</i>")
    else:
        lines.append(f"⚠️ <b>Ca. +{delay} min – Abfahrt ~{hhmm(dep, tz)}</b>")
    # Weicht die eigene Schätzung von der DB ab, beide zeigen. Der Punktwert ist
    # auf etwa +/-2 min genau (MAE 1,78), eine Einzelminute ist nicht belastbar.
    if abs(delay - db_delay) >= 2:
        lines.append(f"<i>DB Navigator sagt +{db_delay} min ({hhmm(est.db_when, tz)}) – hier weichen wir ab.</i>")
    lines += ["", f"🚶 <b>Loslaufen um {hhmm(leave, tz)}</b>", f"({walk_min} min zum Bahnhof)"]

    if extra > 0:
        # Das Regelmodell nennt WELCHER Zug wo kreuzt - das ist seine Stärke und die
        # einzige Begründung, die das ML-Modell nicht liefern kann. Seine berechnete
        # Wartezeit wird NICHT mehr als Minutenzahl gezeigt: im Replay über 2025
        # (ml/replay_2025.py) waren 73 % dieser Blockaden Fehlalarme, der Ist-Median
        # lag bei 2 min. Eine Zahl wie "wartet 15 min" neben einer Prognose von
        # "+2 min" widerspricht sich und schlägt grundlos Alarm. Stattdessen die
        # gemessene Trefferquote.
        lines += ["", "<i>Möglicher Konflikt auf dem eingleisigen Abschnitt:</i>"]
        orte = []
        for h in est.holds:
            for cf in h.conflicts:
                why = "kommt entgegen" if cf.kind == "gegenzug" else "hat Vorrang"
                lines.append(f"• {html.escape(cf.other.label)} {why}")
            orte.append(h.at)
        if orte:
            lines.append(f"• Kreuzungspunkt: {html.escape(', '.join(dict.fromkeys(orte)))}")
        lines += ["", "<i>Aus solchen Lagen werden in etwa jedem vierten Fall "
                      "5 min oder mehr – meistens nicht.</i>"]
    if est.ml_estimate is not None:
        ml_delay = max(0, minutes(est.ml_estimate - est.planned))
        lines.append(f"<i>Modell +{ml_delay} · DB +{db_delay} · typ. Abweichung ±2 min</i>")

    if est.baseline_error >= timedelta(minutes=1):
        lines += ["", f"🔧 Modell sieht schon im Fahrplan +{minutes(est.baseline_error)} min – config prüfen."]
    return "\n".join(lines)


# --------------------------------------------------------------------------- Ausgabe


def send_telegram(token: str, chat_id: str, text: str) -> None:
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
        timeout=15,
    )
    r.raise_for_status()


def alarm_text(est: Estimate, cfg: dict, tz: ZoneInfo) -> str | None:
    """Alarmtext – nur wenn die erwartete Verspätung die Schwelle erreicht."""
    dep, delay = expected_departure(est)
    if delay < int(cfg.get("alarm", {}).get("min_delay_min", 3)):
        return None
    leave = dep - timedelta(minutes=int(cfg["my_train"].get("walk_min", 5)))
    return f"+{delay} min – Abfahrt ~{dep.astimezone(tz):%H:%M}\n🚶 Loslaufen um {leave.astimezone(tz):%H:%M}"


def send_ntfy(topic: str, title: str, text: str, cfg: dict) -> None:
    a = cfg.get("alarm", {})
    r = requests.post(
        a.get("server", "https://ntfy.sh"),
        json={"topic": topic, "title": title, "message": text,
              "priority": int(a.get("priority", 5)), "tags": ["train", "warning"]},
        timeout=15,
    )
    r.raise_for_status()


def alarm(est: Estimate | None, cfg: dict, tz: ZoneInfo, dry_run: bool) -> None:
    """Bei relevanter Verspätung zusätzlich lauter ntfy-Alarm aufs Handy (Env NTFY_TOPIC)."""
    if est is None or not cfg.get("alarm", {}).get("enabled", False):
        return
    text = alarm_text(est, cfg, tz)
    if text is None:
        return
    title = f"{cfg['my_train']['line']} verspätet"
    print(f"\n🚨 Alarm: {title} | {text}")
    topic = os.environ.get("NTFY_TOPIC")
    if dry_run or not topic:
        print("(kein Alarm: --dry-run oder NTFY_TOPIC fehlt)")
        return
    try:
        send_ntfy(topic, title, text, cfg)
        print("→ ntfy-Alarm gesendet")
    except requests.RequestException as exc:  # Telegram ist schon raus – Alarm ist Zugabe
        print(f"ntfy-Alarm fehlgeschlagen: {exc}")


def deliver(text: str, dry_run: bool, strict: bool = False) -> None:
    print(text)
    if dry_run:
        return
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        if strict:
            raise SystemExit("TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID fehlen")
        print("\n(TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID fehlen – nur Konsolenausgabe)")
        return
    send_telegram(token, chat, text)
    print("\n→ an Telegram gesendet")


def log_csv(path: Path, now: datetime, info: dict, tz: ZoneInfo) -> None:
    est: Estimate | None = info.get("est")
    if est is None:
        return
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["datum", "geprueft_um", "soll", "db_prognose", "meine_prognose", "extra_min",
                        "gruende", "tatsaechlich_ab (selbst eintragen)"])
        reasons = " | ".join(cf.other.label for h in est.holds for cf in h.conflicts)
        w.writerow([now.date().isoformat(), hhmm(now, tz), hhmm(est.planned, tz), hhmm(est.db_when, tz),
                    hhmm(est.estimate, tz), minutes(est.extra_vs_db), reasons, ""])


def in_window(now: datetime, sc: dict) -> bool:
    if sc.get("weekdays_only", True) and now.weekday() >= 5:
        return False
    start = time.fromisoformat(sc["window_start"])
    end = time.fromisoformat(sc["window_end"])
    return start <= now.time().replace(tzinfo=None) <= end


def in_recheck_window(now: datetime, sc: dict) -> bool:
    """Zweites, spaeteres Fenster. Naeher an der Abfahrt ist die Prognose besser
    (gemessen 1,66 statt 1,84 min MAE), aber eine zweite Nachricht pro Tag waere
    Laerm - deshalb wird sie nur bei nennenswerter Aenderung verschickt."""
    if not sc.get("recheck_start"):
        return False
    if sc.get("weekdays_only", True) and now.weekday() >= 5:
        return False
    start = time.fromisoformat(sc["recheck_start"])
    end = time.fromisoformat(sc["recheck_end"])
    return start <= now.time().replace(tzinfo=None) <= end


def _state_file() -> Path:
    return HERE / "last_alarm.json"


def letzte_meldung(now: datetime) -> int | None:
    """Verspaetung der heute schon verschickten Nachricht, falls es eine gab."""
    try:
        d = json.loads(_state_file().read_text())
    except (OSError, ValueError):
        return None
    return d.get("delay_min") if d.get("date") == now.date().isoformat() else None


def merke_meldung(now: datetime, delay: int) -> None:
    try:
        _state_file().write_text(json.dumps({"date": now.date().isoformat(),
                                             "delay_min": delay, "at": now.isoformat()}))
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RE19-Morgencheck")
    ap.add_argument("--config", default=str(HERE / "config.toml"))
    ap.add_argument("--dry-run", action="store_true", help="nur ausgeben, nichts senden")
    ap.add_argument("--force", action="store_true", help="Prüffenster ignorieren")
    ap.add_argument("--gate-only", action="store_true", help="nur prüfen, ob jetzt gecheckt werden soll (für GitHub Actions)")
    ap.add_argument("--telegram-test", action="store_true", help="nur eine Testnachricht an Telegram schicken")
    ap.add_argument("--collect", action="store_true", help="Rohdaten + Prognose fürs spätere Modell sammeln (ohne Nachricht)")
    ap.add_argument("--mock", action="store_true", help="kompletter Durchlauf mit erfundenen Daten (ICE-Konflikt)")
    args = ap.parse_args(argv)

    cfg = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    tz = ZoneInfo(cfg.get("timezone", "Europe/Berlin"))
    now = datetime.now(tz)
    # Das Nachcheck-Fenster hat Vorrang, falls sich die beiden ueberlappen. So kann
    # das Hauptfenster grosszuegig bleiben (launchd und GitHub Actions starten mit
    # Verzoegerung), ohne dass der Nachcheck nie greift. Lief der erste Check gar
    # nicht, ist noch keine Meldung gespeichert und der Nachcheck schickt die
    # vollstaendige Nachricht - es entsteht also keine Luecke.
    nach = in_recheck_window(now, cfg["schedule"])
    haupt = (not nach) and in_window(now, cfg["schedule"])
    ok = args.force or args.mock or haupt or nach

    if args.collect:
        import collect

        print(collect.collect_all(cfg, now, make_client(cfg["api"], tz), HERE))
        return 0

    if args.telegram_test:
        deliver(f"✅ re19-watch Test – der Bot kann dir schreiben ({now:%d.%m. %H:%M}).", args.dry_run, strict=True)
        return 0

    if args.gate_only:
        print(f"{now:%a %H:%M} → run={'true' if ok else 'false'}")
        if gh := os.environ.get("GITHUB_OUTPUT"):
            with open(gh, "a") as f:
                f.write(f"run={'true' if ok else 'false'}\n")
        return 0
    if not ok:
        print(f"{now:%a %H:%M}: außerhalb des Prüffensters – nichts zu tun.")
        return 0

    info: dict = {}
    try:
        if args.mock:
            fake_now = now.replace(hour=6, minute=55, second=0, microsecond=0)
            text, info = run_check(cfg, fake_now, client=MockTimetablesClient(tz, fake_now))
            text = "🧪 <i>Testlauf mit erfundenen Daten</i>\n\n" + text
        else:
            text, info = run_check(cfg, now)
    except (requests.RequestException, RuntimeError, KeyError, ValueError) as exc:
        text = f"⚠️ RE19-Check fehlgeschlagen ({type(exc).__name__}): {html.escape(str(exc)[:200])}"

    if (log := cfg.get("logging", {}).get("csv")) and not args.mock:
        log_csv(HERE / log, now, info, tz)
    if info.get("est") and not args.mock and "collect" in cfg:
        import collect

        collect.log_prediction(HERE / cfg["collect"].get("dir", "data"), now, info, "alarm")
    # Beim zweiten Lauf nur melden, wenn sich die Schaetzung nennenswert bewegt hat.
    neu_delay = expected_departure(info["est"])[1] if info.get("est") else None
    if nach and not haupt and not args.force and not args.mock and neu_delay is not None:
        vorher = letzte_meldung(now)
        schwelle = int(cfg["schedule"].get("recheck_min_change_min", 3))
        if vorher is not None and abs(neu_delay - vorher) < schwelle:
            print(f"{now:%H:%M}: Nachcheck – unverändert ({vorher:+d} → {neu_delay:+d} min), "
                  f"keine zweite Nachricht.")
            return 0
        if vorher is not None:
            text = (f"🔄 <b>Aktualisierung</b> (vorher +{vorher} min)\n\n" + text)

    deliver(text, args.dry_run)
    alarm(info.get("est"), cfg, tz, args.dry_run)
    if neu_delay is not None and not args.mock and not args.dry_run:
        merke_meldung(now, neu_delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
