"""
simulate.py – Gleisbelegungs-Simulation fuer einen kurzen Streckenkorridor.

Keine Netzwerkzugriffe, nur Logik -> komplett testbar (siehe test_simulate.py).

Modell (bewusst einfach, fuer ein MVP):
  * Der Korridor ist eine Kette von Punkten (Bahnhoefe, Abzweig), geordnet
    Nord -> Sued. Zwischen zwei Punkten liegt ein Segment mit Fahrzeiten
    je Zugklasse ("regional" inkl. Halt, "fast" fuer ICE/IC ohne Halt).
  * Eingleisige Segmente: Zwei Zuege in GEGENRICHTUNG duerfen sie nicht
    gleichzeitig belegen (+ Puffer fuer Fahrstrassen-Aufloesung).
  * Gleiche Richtung: Mindestabstand (Headway), Ueberholen unmoeglich.
  * Wer Vorrang hat: hoehere Prioritaet (ICE > IC > RE > RB), bei gleicher
    Prioritaet der Zug, der zuerst da ist.
  * Nur MEIN Zug wird verschoben; alle anderen fahren wie von der DB
    prognostiziert. Gewartet wird an Kreuzungspunkten (bzw. am Signal davor).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

FAST = "fast"
REGIONAL = "regional"
ZERO = timedelta(0)


@dataclass
class Corridor:
    points: list[str]  # geografische Reihenfolge Nord -> Sued
    run_min: dict[tuple[str, str], dict[str, float]]  # (a, b) mit a noerdlich von b
    crossing_points: set[str]
    double_track: set[tuple[str, str]] = field(default_factory=set)
    opposing_buffer_min: float = 1.0
    following_headway_min: float = 2.0

    def key(self, a: str, b: str) -> tuple[str, str]:
        return (a, b) if self.points.index(a) < self.points.index(b) else (b, a)

    def run(self, a: str, b: str, cls: str) -> timedelta:
        return timedelta(minutes=self.run_min[self.key(a, b)][cls])

    def is_single(self, a: str, b: str) -> bool:
        return self.key(a, b) not in self.double_track

    def path(self, start: str, end: str) -> list[str]:
        i, j = self.points.index(start), self.points.index(end)
        return self.points[i : j + 1] if i <= j else list(reversed(self.points[j : i + 1]))


@dataclass
class Train:
    label: str
    priority: int
    cls: str
    path: list[str]  # in Fahrtrichtung
    times: dict[str, datetime]  # Abfahrt/Durchfahrt je Punkt (letzter Punkt: Ankunft)
    direction: int  # +1 = nach Sueden, -1 = nach Norden
    trip_id: str | None = None

    def segments(self) -> list[tuple[str, str, datetime, datetime]]:
        return [(a, b, self.times[a], self.times[b]) for a, b in zip(self.path, self.path[1:])]


@dataclass
class Conflict:
    other: Train
    kind: str  # "gegenzug" | "vorrang"
    where: tuple[str, str]
    other_clear: datetime


@dataclass
class Hold:
    at: str
    wait: timedelta
    conflicts: list[Conflict]


@dataclass
class Result:
    times: dict[str, datetime]
    holds: list[Hold]


def make_train(
    c: Corridor,
    label: str,
    priority: int,
    cls: str,
    start: str,
    end: str,
    anchor_point: str,
    anchor_time: datetime,
    trip_id: str | None = None,
) -> Train:
    """Baut einen Zug, dessen Zeit an EINEM Punkt bekannt ist (z.B. Coburg ab 07:03).
    Die Zeiten an allen anderen Punkten werden ueber die Fahrzeiten hochgerechnet."""
    path = c.path(start, end)
    times = {anchor_point: anchor_time}
    i = path.index(anchor_point)
    for k in range(i + 1, len(path)):
        times[path[k]] = times[path[k - 1]] + c.run(path[k - 1], path[k], cls)
    for k in range(i - 1, -1, -1):
        times[path[k]] = times[path[k + 1]] - c.run(path[k], path[k + 1], cls)
    direction = 1 if c.points.index(end) > c.points.index(start) else -1
    return Train(label, priority, cls, path, times, direction, trip_id)


def blocks(path: list[str], crossing: set[str]) -> list[list[str]]:
    """Teilt einen Laufweg an Kreuzungspunkten in Bloecke, in denen nicht gewartet
    bzw. nicht gekreuzt werden kann."""
    out: list[list[str]] = []
    cur = [path[0]]
    for p in path[1:]:
        cur.append(p)
        if p in crossing and p != path[-1]:
            out.append(cur)
            cur = [p]
    if len(cur) > 1:
        out.append(cur)
    return out


def _propagate(block: list[str], start: datetime, cls: str, c: Corridor) -> dict[str, datetime]:
    sched = {block[0]: start}
    for a, b in zip(block, block[1:]):
        sched[b] = sched[a] + c.run(a, b, cls)
    return sched


def _has_precedence(o: Train, me: Train, o_t: datetime, my_t: datetime) -> bool:
    if o.priority != me.priority:
        return o.priority > me.priority
    return o_t <= my_t  # gleicher Rang: wer zuerst kommt


def _required_shift(
    sched: dict[str, datetime], block: list[str], me: Train, others: list[Train], c: Corridor
) -> tuple[timedelta, list[Conflict]]:
    mine = {c.key(a, b): (sched[a], sched[b]) for a, b in zip(block, block[1:])}
    buf = timedelta(minutes=c.opposing_buffer_min)
    head = timedelta(minutes=c.following_headway_min)
    shift, found = ZERO, []

    for o in others:
        shared = [s for s in o.segments() if c.key(s[0], s[1]) in mine]
        if not shared:
            continue

        if o.direction != me.direction:
            single = [s for s in shared if c.is_single(s[0], s[1])]
            if not single:
                continue
            o_in, o_out = min(s[2] for s in single), max(s[3] for s in single)
            my_iv = [mine[c.key(s[0], s[1])] for s in single]
            m_in, m_out = min(x[0] for x in my_iv), max(x[1] for x in my_iv)
            if not _has_precedence(o, me, o_in, m_in):
                continue
            if m_in < o_out + buf and o_in < m_out + buf:
                need = o_out + buf - m_in
                if need > ZERO:
                    shift = max(shift, need)
                    found.append(Conflict(o, "gegenzug", (single[0][0], single[-1][1]), o_out))
        else:
            for a, b, ta, tb in shared:
                m, x = mine[c.key(a, b)]
                if not _has_precedence(o, me, ta, m):
                    continue
                if m < tb + head and ta < x + head:
                    need = max(ta + head - m, tb + head - x)
                    if need > ZERO:
                        shift = max(shift, need)
                        found.append(Conflict(o, "vorrang", (a, b), tb))
                        break
    return shift, found


def resolve(me: Train, others: list[Train], c: Corridor, max_iter: int = 50) -> Result:
    """Schiebt MEINEN Zug so lange nach hinten, bis er keinen Zug mit Vorrang mehr stoert."""
    others = [o for o in others if o is not me]
    t = me.times[me.path[0]]
    times = {me.path[0]: t}
    holds: list[Hold] = []

    for block in blocks(me.path, c.crossing_points):
        start, conflicts = t, []
        sched = _propagate(block, start, me.cls, c)
        for _ in range(max_iter):
            shift, found = _required_shift(sched, block, me, others, c)
            if shift <= ZERO:
                break
            conflicts.extend(found)
            start += shift
            sched = _propagate(block, start, me.cls, c)
        if start > t:
            seen, uniq = set(), []
            for cf in conflicts:
                k = (cf.other.label, cf.kind)
                if k not in seen:
                    seen.add(k)
                    uniq.append(cf)
            holds.append(Hold(block[0], start - t, uniq))
        times.update(sched)
        t = sched[block[-1]]
    return Result(times, holds)


@dataclass
class Estimate:
    planned: datetime
    db_when: datetime
    sim_plan: datetime
    sim_prog: datetime
    baseline_error: timedelta  # >0 heisst: Modell sieht schon im PLAN einen Konflikt -> kalibrieren
    estimate: datetime
    extra_vs_db: timedelta
    holds: list[Hold]


def estimate(
    me_plan: Train,
    others_plan: list[Train],
    me_prog: Train,
    others_prog: list[Train],
    c: Corridor,
    point: str,
    planned: datetime,
    db_when: datetime,
) -> Estimate:
    """Zweimal simulieren: mit Soll-Zeiten (Kalibrier-Basislinie) und mit aktuellen
    Prognosen. Nur die DIFFERENZ zaehlt – so werden planmaessige Kreuzungen,
    die schon im Fahrplan eingerechnet sind, nicht faelschlich als Verspaetung gemeldet."""
    r_plan = resolve(me_plan, others_plan, c)
    r_prog = resolve(me_prog, others_prog, c)
    sim_plan, sim_prog = r_plan.times[point], r_prog.times[point]
    baseline = max(sim_plan - planned, ZERO)
    est = max(db_when, sim_prog - baseline)
    return Estimate(planned, db_when, sim_plan, sim_prog, baseline, est, est - db_when, r_prog.holds)
