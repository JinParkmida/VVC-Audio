"""Minimal Standard MIDI File (SMF) reader/writer.

Only what VVC needs: note on/off, control change, pitch bend, program change
and meta events. Writes running status; uses note-on velocity 0 as note-off
so running status survives across note events.
"""
from __future__ import annotations

import struct


def _vlq(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def _read_vlq(data: bytes, i: int) -> tuple[int, int]:
    n = 0
    while True:
        b = data[i]
        i += 1
        n = (n << 7) | (b & 0x7F)
        if not b & 0x80:
            return n, i


class Track:
    def __init__(self, name: str | None = None):
        self.events: list[tuple[int, int, bytes]] = []
        if name:
            self.meta(0, 0x03, name.encode("utf-8"))

    def _add(self, tick: int, raw: bytes) -> None:
        # Insertion order breaks ties at equal ticks, so callers control
        # e.g. note-off-before-note-on at the same instant.
        self.events.append((int(tick), len(self.events), raw))

    def meta(self, tick: int, typ: int, payload: bytes) -> None:
        self._add(tick, bytes([0xFF, typ]) + _vlq(len(payload)) + payload)

    def text(self, tick: int, s: str) -> None:
        self.meta(tick, 0x01, s.encode("utf-8"))

    def tempo(self, tick: int, us_per_quarter: int) -> None:
        self.meta(tick, 0x51, us_per_quarter.to_bytes(3, "big"))

    def note_on(self, tick: int, ch: int, note: int, vel: int) -> None:
        self._add(tick, bytes([0x90 | ch, note, max(1, min(127, vel))]))

    def note_off(self, tick: int, ch: int, note: int) -> None:
        self._add(tick, bytes([0x90 | ch, note, 0]))

    def cc(self, tick: int, ch: int, num: int, val: int) -> None:
        self._add(tick, bytes([0xB0 | ch, num, max(0, min(127, val))]))

    def pitch_bend(self, tick: int, ch: int, value14: int) -> None:
        v = max(0, min(16383, value14))
        self._add(tick, bytes([0xE0 | ch, v & 0x7F, v >> 7]))

    def program(self, tick: int, ch: int, prog: int) -> None:
        self._add(tick, bytes([0xC0 | ch, prog]))

    def encode(self) -> bytes:
        out = bytearray()
        last = 0
        running = None
        for tick, _, raw in sorted(self.events, key=lambda e: (e[0], e[1])):
            out += _vlq(tick - last)
            last = tick
            st = raw[0]
            if st < 0xF0:
                if st == running:
                    out += raw[1:]
                else:
                    out += raw
                    running = st
            else:
                out += raw
                running = None
        out += _vlq(0) + b"\xFF\x2F\x00"
        return b"MTrk" + struct.pack(">I", len(out)) + bytes(out)


def write_smf(path: str, tracks: list[Track], ppq: int) -> int:
    body = b"".join(t.encode() for t in tracks)
    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), ppq)
    data = header + body
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def read_smf(path: str):
    """Return (ppq, tracks). Each track is a list of
    (abs_tick, kind, status_or_metatype, payload_bytes) with kind 'meta' or 'midi'."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"MThd":
        raise ValueError("not a Standard MIDI File")
    hlen = struct.unpack(">I", data[4:8])[0]
    _fmt, ntrk, ppq = struct.unpack(">HHH", data[8:14])
    if ppq & 0x8000:
        raise ValueError("SMPTE time division not supported")
    i = 8 + hlen
    tracks = []
    for _ in range(ntrk):
        if data[i:i + 4] != b"MTrk":
            raise ValueError("bad track chunk")
        ln = struct.unpack(">I", data[i + 4:i + 8])[0]
        j, end = i + 8, i + 8 + ln
        tick, running, evs = 0, None, []
        while j < end:
            d, j = _read_vlq(data, j)
            tick += d
            st = data[j]
            if st == 0xFF:
                typ = data[j + 1]
                n, j = _read_vlq(data, j + 2)
                evs.append((tick, "meta", typ, bytes(data[j:j + n])))
                j += n
                if typ == 0x2F:
                    break
            elif st in (0xF0, 0xF7):
                n, j = _read_vlq(data, j + 1)
                j += n
            else:
                if st & 0x80:
                    running = st
                    j += 1
                else:
                    st = running
                n = 1 if (st & 0xF0) in (0xC0, 0xD0) else 2
                evs.append((tick, "midi", st, bytes(data[j:j + n])))
                j += n
        tracks.append(evs)
        i = end
    return ppq, tracks
