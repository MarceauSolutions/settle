#!/usr/bin/env python3
"""Render Settle's tracks to audio/*.m4a. Needs numpy, scipy, ffmpeg.

Each track fades in over FADE_IN seconds and out over the last TAIL seconds.
The page loops the file natively, and the sleep timer seeks into the tail.
TAIL and BREATH_SECONDS must match index.html.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import signal

RATE = 44100
DURATION = 30 * 60
BLOCK = 10
FADE_IN = 8
TAIL = 60
BREATH_SECONDS = 10.9  # 5.5 breaths per minute
INHALE_SHARE = 0.41
TONE_DB = -24
AMBIENCE_DB = -21
OUT = Path(__file__).resolve().parent.parent / "audio"

TRACKS = {
    "deep-sleep": {"ambience": "brown", "carrier": 200, "beat": 2},
    "drift-off": {"ambience": "rain", "carrier": 200, "beat": 4},
    "exhale": {"ambience": "ocean", "carrier": 200, "beat": 6},
    "unwind": {"ambience": "wind", "carrier": 240, "beat": 10},
    "rain": {"ambience": "rain"},
    "ocean": {"ambience": "ocean"},
}

PINK_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
PINK_A = [1, -2.494956002, 2.017265875, -0.522189400]


class Filter:
    """Stateful IIR filter, one state per channel, so blocks join seamlessly."""

    def __init__(self, b, a):
        self.b, self.a = b, a
        self.state = np.zeros((2, max(len(a), len(b)) - 1))

    def __call__(self, x):
        y, self.state = signal.lfilter(self.b, self.a, x, axis=1, zi=self.state)
        return y


def butter(order, cutoff, kind):
    return Filter(*signal.butter(order, cutoff, kind, fs=RATE))


def breath(t, offset=0.0):
    """Swell that rises for the inhale and falls for the exhale, 0 to 1."""
    p = ((t + offset) / BREATH_SECONDS) % 1
    rising = np.sin(np.pi / 2 * p / INHALE_SHARE) ** 2
    falling = np.cos(np.pi / 2 * (p - INHALE_SHARE) / (1 - INHALE_SHARE)) ** 2
    return np.where(p < INHALE_SHARE, rising, falling)


class Brown:
    def __init__(self):
        self.integrate = Filter([1], [1, -0.995])
        self.low = butter(2, 500, "low")

    def __call__(self, rng, t):
        return self.low(self.integrate(rng.standard_normal((2, t.size))))


class Rain:
    def __init__(self):
        self.pink = Filter(PINK_B, PINK_A)
        self.body = butter(2, [500, 9000], "bandpass")
        self.ring = Filter([1], [1, -0.97])
        self.drops = butter(2, 1800, "high")

    def __call__(self, rng, t):
        wash = self.body(self.pink(rng.standard_normal((2, t.size))))
        hits = (rng.random((2, t.size)) < 45 / RATE) * rng.lognormal(0, 0.8, (2, t.size))
        drops = self.drops(self.ring(hits * rng.standard_normal((2, t.size))))
        swell = 0.85 + 0.15 * np.sin(2 * np.pi * t / 47)
        drops = 0.6 * np.tanh(drops / max(3 * drops.std(), 1e-9))  # soft ceiling, no pops
        return (wash / wash.std() + drops) * swell


class Ocean:
    def __init__(self):
        self.integrate = Filter([1], [1, -0.99])
        self.low = butter(2, 900, "low")
        self.hiss = butter(2, 1500, "high")

    def __call__(self, rng, t):
        roar = self.low(self.integrate(rng.standard_normal((2, t.size))))
        hiss = self.hiss(rng.standard_normal((2, t.size)))
        out = np.empty((2, t.size))
        for ch, offset in enumerate((0.0, -0.35)):
            swell = breath(t, offset)
            out[ch] = (
                roar[ch] / roar[ch].std() * (0.3 + 0.7 * swell)
                + hiss[ch] / hiss[ch].std() * 0.35 * swell**2
            )
        return out


class Wind:
    def __init__(self):
        self.pink = Filter(PINK_B, PINK_A)
        self.band = butter(2, [180, 1100], "bandpass")

    def __call__(self, rng, t):
        x = self.band(self.pink(rng.standard_normal((2, t.size))))
        gust = 0.65 + 0.2 * np.sin(2 * np.pi * t / 23) + 0.15 * np.sin(2 * np.pi * t / 61)
        pan = 0.12 * np.sin(2 * np.pi * t / 37)
        return x / x.std() * gust * np.vstack((1 + pan, 1 - pan))


AMBIENCE = {"brown": Brown, "rain": Rain, "ocean": Ocean, "wind": Wind}


def render(name, spec):
    print("rendering", name, flush=True)
    rng = np.random.default_rng(sum(name.encode()))
    ambience = AMBIENCE[spec["ambience"]]()
    ambience_gain = 10 ** (AMBIENCE_DB / 20)
    tone_gain = 10 ** (TONE_DB / 20) * np.sqrt(2)
    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "f32le", "-ar", str(RATE), "-ac", "2", "-i", "-",
         "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(OUT / f"{name}.m4a")],
        stdin=subprocess.PIPE,
    )
    level = None
    for start in range(0, DURATION, BLOCK):
        t = np.arange(start * RATE, (start + BLOCK) * RATE) / RATE
        x = ambience(rng, t)
        if level is None:
            level = x.std()  # one fixed scale, so the swells are not flattened
        x = x / level * ambience_gain
        if "beat" in spec:
            x[0] += tone_gain * np.sin(2 * np.pi * spec["carrier"] * t)
            x[1] += tone_gain * np.sin(2 * np.pi * (spec["carrier"] + spec["beat"]) * t)
        fade = np.minimum(np.clip(t / FADE_IN, 0, 1), np.clip((DURATION - t) / TAIL, 0, 1))
        x = np.tanh(x * np.sin(np.pi / 2 * fade) ** 2)
        encoder.stdin.write(x.T.astype("<f4").tobytes())
    encoder.stdin.close()
    if encoder.wait():
        sys.exit(f"ffmpeg failed on {name}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name in sys.argv[1:] or TRACKS:
        render(name, TRACKS[name])
