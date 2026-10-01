#!/usr/bin/env python3
"""
collect.py – Datensammler für ein späteres, gelerntes Prognosemodell.

Läuft alle paar Minuten (launchd) und speichert:
  data/YYYY-MM-DD/raw/        Rohantworten der Timetables-API (gzip, unverändert)
                              plan_<eva>_<HH>.xml.gz  einmal je Stunde (Soll-Fahrplan)
                              fchg_<eva>_<HHMMSS>.xml.gz  jeder Lauf (Echtzeit-Stand)
  data/YYYY-MM-DD/plan.jsonl  Soll-Halte, flach (eine Zeile je Halt und Ankunft/Abfahrt)
  data/YYYY-MM-DD/changes.jsonl  Echtzeit-Stand (Prognosezeit, Ausfall, Gleis, Meldungen/Gründe) –
                              nur Einträge, die sich seit dem letzten Lauf geändert haben
  data/YYYY-MM-DD/predictions.jsonl  Modell- und DB-Prognose für meinen Zug im Vorhersagefenster

Tatsächliche Zeiten ergeben sich später aus dem letzten Echtzeit-Stand nach Abfahrt
(changes.jsonl), Vorlaufzeiten aus den Zeitstempeln der Läufe.
"""

from __future__ import annotations

import gzip
import json
import xml.etree.ElementTree as ET
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

import re19watch as rw


def _iso(s: str | None, tz: ZoneInfo) -> str | None:
    t = rw._tt_time(s, tz)
    return t.isoformat() if t else None


def _msgs(el: ET.Element | None) -> list[dict]:
    """<m>-Meldungen: t=d Verspätungsgrund (c=Code), t=f Qualität, t=h HIM-Info, t=q ..."""
    if el is None:
        return []
    return [{k: m.get(k) for k in ("id", "t", "c", "cat", "ts", "pr") if m.get(k)} for m in el.findall("m")]


def plan_rows(root: ET.Element, eva: str, tz: ZoneInfo) -> list[dict]:
    rows = []
    for s in root:
        tl = s.find("tl")
        tla = tl.attrib if tl is not None else {}
        for tag, kind in (("ar", "arrival"), ("dp", "departure")):
            e = s.find(tag)
            if e is None:
                continue
            rows.append({
                "eva": eva, "stop_id": s.get("id"), "trip_id": s.get("id", "").rsplit("-", 1)[0],
                "kind": kind, "category": tla.get("c"), "number": tla.get("n"), "operator": tla.get("o"),
                "line": e.get("l"), "planned": _iso(e.get("pt"), tz), "platform": e.get("pp"),
                "path": (e.get("ppth") or "").split("|") if e.get("ppth") else [],
            })
    return rows


def change_rows(root: ET.Element, eva: str, snap: datetime, tz: ZoneInfo) -> list[dict]:
    rows = []
    for s in root:
        stop_msgs = _msgs(s)
        for tag, kind in (("ar", "arrival"), ("dp", "departure")):
            e = s.find(tag)
            if e is None:
                continue
            rows.append({
                "snapshot": snap.isoformat(), "eva": eva, "stop_id": s.get("id"),
                "trip_id": s.get("id", "").rsplit("-", 1)[0], "kind": kind,
                "changed": _iso(e.get("ct"), tz), "status": e.get("cs"), "cancel_time": _iso(e.get("clt"), tz),
                "platform": e.get("cp"), "path": e.get("cpth").split("|") if e.get("cpth") else None,
                "messages": _msgs(e), "stop_messages": stop_msgs,
            })
    return rows


def _is_new(row: dict, seen: dict, now_seen: dict) -> bool:
    """True, wenn sich der Eintrag seit dem letzten Lauf geändert hat; merkt ihn für den nächsten Lauf."""
    key = f"{row['eva']}|{row['stop_id']}|{row['kind']}"
    val = json.dumps({k: v for k, v in row.items() if k != "snapshot"}, sort_keys=True)
    now_seen[key] = val
    return seen.get(key) != val


def _append(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _fetch_raw(client: rw.TimetablesClient, path: str) -> bytes | None:
    r = requests.get(f"{rw.TT_BASE}/{path}", headers=client.headers, timeout=client.timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    client._cache[path] = ET.fromstring(r.content)  # gleiche Daten gleich für die Prognose nutzen
    return r.content


def prediction_record(now: datetime, info: dict, source: str) -> dict:
    est, me = info["est"], info["me"]
    dep, delay = rw.expected_departure(est)
    return {
        "snapshot": now.isoformat(), "source": source, "trip_id": me.trip_id, "line": me.line,
        "planned": est.planned.isoformat(), "db_when": est.db_when.isoformat(),
        "sim_plan": est.sim_plan.isoformat(), "sim_prog": est.sim_prog.isoformat(),
        "model_estimate": est.estimate.isoformat(), "expected": dep.isoformat(), "expected_delay_min": delay,
        "extra_vs_db_min": est.extra_vs_db.total_seconds() / 60,
        "baseline_error_min": est.baseline_error.total_seconds() / 60,
        "holds": [{
            "at": h.at, "wait_min": h.wait.total_seconds() / 60,
            "conflicts": [{"with": cf.other.label, "kind": cf.kind, "where": list(cf.where),
                           "other_priority": cf.other.priority, "other_clear": cf.other_clear.isoformat()}
                          for cf in h.conflicts],
        } for h in est.holds],
    }


def log_prediction(data_dir: Path, now: datetime, info: dict, source: str) -> None:
    if info.get("est") is None:
        return
    day = data_dir / f"{now:%Y-%m-%d}"
    day.mkdir(parents=True, exist_ok=True)
    _append(day / "predictions.jsonl", [prediction_record(now, info, source)])


def collect_all(cfg: dict, now: datetime, client: rw.TimetablesClient, base: Path) -> str:
    """Alle Sammel-Profile: [collect] plus [collect_<name>] (je eigener Ordner, eigene Stationen)."""
    out = []
    for key, cc in cfg.items():
        if key == "collect" or key.startswith("collect_"):
            out.append(f"[{key}] " + collect(cfg, now, client, base / cc.get("dir", "data"), cc))
    return "\n".join(out)


def collect(cfg: dict, now: datetime, client: rw.TimetablesClient, data_dir: Path, cc: dict | None = None) -> str:
    tz = now.tzinfo
    cc = cc or cfg["collect"]
    if cc.get("until_date") and now.date().isoformat() > cc["until_date"]:
        return "abgelaufen (until_date)"
    wf, wu = (time.fromisoformat(cc.get(k, d)) for k, d in (("window_from", "00:00"), ("window_until", "23:59")))
    if cc.get("full_day_until") and now.date().isoformat() <= cc["full_day_until"]:
        wf, wu = time(0, 0), time(23, 59, 59)  # Test-Tage: rund um die Uhr
    if (cc.get("weekdays_only") and now.weekday() >= 5) or not wf <= now.time().replace(tzinfo=None) <= wu:
        return f"{now:%a %H:%M}: außerhalb des Sammelfensters"
    day = data_dir / f"{now:%Y-%m-%d}"
    raw = day / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    n_plan = n_chg = 0
    state = data_dir / "last_changes.json"
    try:
        seen = json.loads(state.read_text())
    except (OSError, ValueError):
        seen = {}
    now_seen: dict = {}  # nur aktueller Stand, damit die Datei nicht endlos wächst

    for st in cc["stations"]:
        eva = str(st["eva"])
        # Soll-Plan: je Stunde einmal (Stunde davor bis lookahead)
        h0 = now.replace(minute=0, second=0, microsecond=0)
        for i in range(-1, int(cc.get("plan_lookahead_h", 3)) + 1):
            h = h0 + timedelta(hours=i)
            f = data_dir / f"{h:%Y-%m-%d}" / "raw" / f"plan_{eva}_{h:%H}.xml.gz"
            if f.exists():
                continue
            content = _fetch_raw(client, f"plan/{eva}/{h:%y%m%d}/{h:%H}")
            if content is None:
                continue
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(gzip.compress(content))
            _append(f.parent.parent / "plan.jsonl", plan_rows(ET.fromstring(content), eva, tz))
            n_plan += 1
        # Echtzeit: jeder Lauf
        content = _fetch_raw(client, f"fchg/{eva}")
        if content is not None:
            (raw / f"fchg_{eva}_{now:%H%M%S}.xml.gz").write_bytes(gzip.compress(content))
            rows = [r for r in change_rows(ET.fromstring(content), eva, now, tz) if _is_new(r, seen, now_seen)]
            _append(day / "changes.jsonl", rows)
            n_chg += len(rows)

    state.write_text(json.dumps(now_seen))

    # Prognose für meinen Zug mitschreiben (ohne Nachricht), solange es spannend ist
    msg = f"{now:%H:%M} gesammelt: {n_plan} Pläne, {n_chg} Echtzeit-Einträge"
    if "predict_from" not in cc:
        return msg
    start, end = (time.fromisoformat(cc[k]) for k in ("predict_from", "predict_until"))
    if start <= now.time().replace(tzinfo=None) <= end:
        try:
            _, info = rw.run_check(cfg, now, client=client)
            log_prediction(data_dir, now, info, "collect")
            msg += ", Prognose geloggt" if info else ", Zug nicht gefunden"
        except (requests.RequestException, RuntimeError, KeyError, ValueError) as exc:
            msg += f", Prognose fehlgeschlagen: {exc}"
    return msg
