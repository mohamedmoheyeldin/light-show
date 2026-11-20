"""Generate a Tesla light show (.fseq, V2 uncompressed, 200 channels) straight from an audio file.

The show is choreographed from the music itself: a beat grid (from the BPM, phase-locked to the kick
drum), kick / snare / hi-hat onsets and a per-bar intensity level that picks how busy the lights are.
It is tuned for Model 3/Y lights (ramping signature, combined channels 4-6, OR'd aux park + side markers)
and plays on every supported Tesla.

Usage:
    python tools/auto_show.py "song.wav" OUTPUT_DIR [--bpm 126] [--name lightshow]

OUTPUT_DIR/LightShow/<name>.fseq and <name>.wav are written (audio re-encoded to 44.1 kHz 16-bit wav).
Needs numpy and ffmpeg on PATH.
"""
import argparse
import os
import struct
import subprocess
import sys

import numpy as np

SR = 44100
STEP_MS = 20
HOP = SR * STEP_MS // 1000  # 882 samples = one frame
NCH = 200

# xLights channel numbers (1-based) from tesla_xlights_show_folder
L_OUTER, R_OUTER, L_INNER, R_INNER = 1, 2, 3, 4
L_SIG, R_SIG = 5, 6
L_CH4, R_CH4, L_CH5, R_CH5, L_CH6, R_CH6 = 7, 8, 9, 10, 11, 12
L_FTURN, R_FTURN = 13, 14
L_FOG, R_FOG = 15, 16
L_AUX, R_AUX = 17, 18
L_MARK, R_MARK = 19, 20
L_REP, R_REP = 21, 22
L_RTURN, R_RTURN = 23, 24
BRAKE, L_TAIL, R_TAIL, REVERSE, REAR_FOG, PLATE = 25, 26, 27, 28, 29, 30
L_MIRROR, R_MIRROR = 35, 36
WINDOWS = (37, 38, 39, 40)
LIFTGATE, CHARGE_PORT = 41, 46
RGB_ZONES = (176, 179, 182, 185, 188, 191)  # front display + 5 accent zones, 3 channels each

# Values (xLights effect brightness -> byte)
ON = 255
ON_500, ON_1000, ON_2000 = 178, 204, 229
OFF_500, OFF_1000, OFF_2000 = 25, 51, 76
OPEN, DANCE, CLOSE = 63, 127, 191


def load_audio(path):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def analyse(x, nframes):
    win = 2048
    pad = np.concatenate([np.zeros(win // 2, np.float32), x, np.zeros(win, np.float32)])
    idx = np.arange(nframes)[:, None] * HOP + np.arange(win)[None, :]
    frames = pad[idx] * np.hanning(win).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(win, 1 / SR)

    def band(lo, hi):
        b = np.log1p(mag[:, (freqs >= lo) & (freqs < hi)] * 10)
        flux = np.maximum(np.diff(b, axis=0, prepend=b[:1]), 0).sum(axis=1)
        return flux / (np.percentile(flux, 99) + 1e-9)

    rms = np.sqrt((frames ** 2).mean(axis=1))
    return {
        "kick": band(35, 160),
        "snare": band(1500, 5000),
        "hat": band(6000, 16000),
        "rms": rms / (rms.max() + 1e-9),
    }


def peaks(sig, thresh, min_gap):
    """Local maxima above an adaptive threshold, at least min_gap frames apart."""
    k = 25
    local = np.convolve(sig, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    out, last = [], -10 ** 9
    for i in range(1, len(sig) - 1):
        if sig[i] >= sig[i - 1] and sig[i] > sig[i + 1] and sig[i] > thresh and sig[i] > 1.5 * local[i]:
            if i - last >= min_gap:
                out.append(i)
                last = i
    return np.array(out, dtype=int)


def beat_grid(kick, nframes, bpm):
    period = 60000.0 / bpm / STEP_MS  # frames per beat
    best, best_phase = -1, 0.0
    for phase in np.arange(0, period, 0.25):
        pos = np.round(np.arange(phase, nframes, period)).astype(int)
        pos = pos[pos < nframes]
        score = kick[pos].sum()
        if score > best:
            best, best_phase = score, phase
    beats = np.round(np.arange(best_phase, nframes, period)).astype(int)
    return beats[beats < nframes], period


class Show:
    def __init__(self, nframes):
        self.d = np.zeros((nframes, NCH), dtype=np.uint8)
        self.n = nframes
        self.closure_cmds = {}

    def set(self, ch, start, length, value):
        s, e = max(0, int(start)), min(self.n, int(start + length))
        if e > s:
            self.d[s:e, ch - 1] = value

    def flash(self, chs, start, length, value=ON):
        for ch in chs:
            self.set(ch, start, length, value)

    def closure(self, ch, start, length, value):
        if value in (OPEN, DANCE, CLOSE):
            self.closure_cmds[ch] = self.closure_cmds.get(ch, 0) + 1
        self.set(ch, start, length, value)

    def rgb(self, start, length, color, zones=RGB_ZONES):
        for z in zones:
            for i, c in enumerate(color):
                self.set(z + i, start, length, c)


PALETTE = [(70, 0, 255), (0, 90, 255), (160, 0, 255), (0, 200, 255), (255, 0, 140)]


def build(feat, nframes, bpm):
    show = Show(nframes)
    beats, period = beat_grid(feat["kick"], nframes, bpm)
    bar = 4
    nbars = int(np.ceil(len(beats) / bar))

    # Intensity per bar from loudness and kick density -> level 0..3
    energy = []
    for b in range(nbars):
        s = beats[b * bar]
        e = beats[(b + 1) * bar] if (b + 1) * bar < len(beats) else nframes
        energy.append(feat["rms"][s:e].mean() * 0.6 + feat["kick"][s:e].mean() * 0.4)
    energy = np.array(energy)
    smooth = np.convolve(energy, np.ones(3) / 3, mode="same")
    q = np.quantile(smooth, [0.25, 0.5, 0.8])
    level = np.digitize(smooth, q)  # 0..3
    # sections are 4-bar phrases: take the phrase median so the look doesn't flicker bar to bar
    for p in range(0, nbars, 4):
        level[p:p + 4] = int(np.median(level[p:p + 4]))

    kicks = peaks(feat["kick"], 0.35, int(period * 0.4))
    snares = peaks(feat["snare"], 0.35, int(period * 0.4))
    hats = peaks(feat["hat"], 0.3, int(period * 0.2))
    fl = max(5, int(period * 0.3))  # flash length (~140 ms at 126 BPM)

    def in_bar(arr, s, e):
        return arr[(arr >= s) & (arr < e)]

    for b in range(nbars):
        s = beats[b * bar]
        e = beats[(b + 1) * bar] if (b + 1) * bar < len(beats) else nframes
        lv = level[b]
        color = PALETTE[(b // 4) % len(PALETTE)]
        bb = beats[b * bar:(b + 1) * bar]

        if lv == 0:
            # Calm: signature + channels 4-6 breathe once per bar, interior glows
            half = (e - s) // 2
            show.flash([L_SIG, R_SIG, L_CH4, R_CH4], s, half, ON_1000)
            show.flash([L_SIG, R_SIG, L_CH4, R_CH4], s + half, e - s - half, OFF_1000)
            show.rgb(s, e - s, tuple(c // 3 for c in color))
            for k in in_bar(kicks, s, e):
                show.flash([L_TAIL, R_TAIL], k, fl)
        elif lv == 1:
            show.flash([L_SIG, R_SIG], s, e - s, ON)
            show.rgb(s, e - s, tuple(c // 2 for c in color))
            for i, k in enumerate(in_bar(kicks, s, e)):
                show.flash([L_INNER, R_INNER, L_TAIL, R_TAIL], k, fl)
            for i, sn in enumerate(in_bar(snares, s, e)):
                side = (L_FTURN, L_RTURN) if i % 2 == 0 else (R_FTURN, R_RTURN)
                show.flash(side, sn, fl)
        else:
            # Busy: beams on the kick, L/R alternation per beat, hats on the markers
            show.rgb(s, e - s, color)
            for i, beat in enumerate(bb):
                left = (b * bar + i) % 2 == 0
                sig = (L_SIG, L_CH4, L_CH5, L_CH6) if left else (R_SIG, R_CH4, R_CH5, R_CH6)
                show.flash(sig, beat, int(period * 0.6))
                show.flash([L_REP if left else R_REP], beat, fl)
            for k in in_bar(kicks, s, e):
                show.flash([L_INNER, R_INNER, L_OUTER, R_OUTER, BRAKE, L_TAIL, R_TAIL], k, fl)
                show.rgb(k, fl, (255, 255, 255), zones=(176,))
            for i, sn in enumerate(in_bar(snares, s, e)):
                show.flash([L_FTURN, R_FTURN, L_RTURN, R_RTURN, PLATE], sn, fl)
                if lv == 3:
                    show.flash([L_FOG, R_FOG, REVERSE, REAR_FOG], sn, fl)
            for i, h in enumerate(in_bar(hats, s, e)):
                if not show.d[h, L_AUX - 1]:
                    show.flash([L_AUX, L_MARK] if i % 2 == 0 else [R_AUX, R_MARK], h, max(4, fl // 2))

    # Opening: signature ramps up over the first two seconds before the music gets going
    show.flash([L_SIG, R_SIG], 0, min(100, beats[0] + 100), ON_2000)

    # Closures, kept well inside Tesla's per-show limits
    hot = [b for b in range(nbars) if level[b] == 3]
    if hot:
        # biggest section = longest run of level-3 bars
        runs, cur = [], [hot[0]]
        for b in hot[1:]:
            if b == cur[-1] + 1:
                cur.append(b)
            else:
                runs.append(cur)
                cur = [b]
        runs.append(cur)
        main = max(runs, key=len)
        ds = beats[main[0] * bar]
        de = beats[(main[-1] + 1) * bar] if (main[-1] + 1) * bar < len(beats) else nframes - 1
        dance_len = min(de - ds, 1500)  # <= 30 s of dancing
        # charge port: open 2 s before the drop, rainbow dance through it, close after
        show.closure(CHARGE_PORT, ds - 150, 100, OPEN)
        show.closure(CHARGE_PORT, ds, dance_len, DANCE)
        show.closure(CHARGE_PORT, ds + dance_len, 50, CLOSE)
        # windows dance with the drop (dance does not need them open first)
        for w in WINDOWS:
            show.closure(w, ds, min(dance_len, 750), DANCE)
        # trunk (power trunk only): open early in the run-up, dance, close after
        if ds > 900:
            show.closure(LIFTGATE, ds - 850, 50, OPEN)
            show.closure(LIFTGATE, ds, min(dance_len, 750), DANCE)
            show.closure(LIFTGATE, ds + min(dance_len, 750) + 50, 50, CLOSE)
    # mirrors fold in and out on every level-3 phrase start (<= 20 commands)
    starts = [b for b in range(nbars) if level[b] == 3 and (b == 0 or level[b - 1] != 3)]
    for b in starts[:9]:
        t = beats[b * bar]
        show.closure(L_MIRROR, t, 50, CLOSE)
        show.closure(R_MIRROR, t, 50, CLOSE)
        show.closure(L_MIRROR, t + int(period * 8), 50, OPEN)
        show.closure(R_MIRROR, t + int(period * 8), 50, OPEN)

    # Ending: everything off, signature fades out over 2 s
    tail = nframes - 120
    show.d[tail:, :] = 0
    show.flash([L_SIG, R_SIG], tail - 60, 60, ON)
    show.flash([L_SIG, R_SIG], tail, 110, OFF_2000)
    return show, beats, level


def write_fseq(path, data, media_name):
    nframes, nch = data.shape

    def var(code, text):
        b = code.encode() + text.encode() + b"\0"
        return struct.pack("<H", len(b) + 2) + b

    vh = var("mf", media_name) + var("sp", "tesla auto_show.py")
    start = 32 + len(vh)
    start += (-start) % 4
    header = b"PSEQ" + struct.pack("<HBBHIIBBBBBBQ", start, 0, 2, 32, nch, nframes, STEP_MS,
                                   0, 0, 0, 0, 0, 0)
    header = header + vh
    header += b"\0" * (start - len(header))
    with open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("outdir")
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--name", default="lightshow")
    a = ap.parse_args()

    x = load_audio(a.audio)
    nframes = int(np.ceil(len(x) / HOP))
    feat = analyse(x, nframes)
    bpm = a.bpm
    if bpm is None:  # autocorrelation of the kick flux, 80-180 BPM
        k = feat["kick"] - feat["kick"].mean()
        ac = np.correlate(k, k, mode="full")[len(k) - 1:]
        lags = np.arange(int(60000 / 180 / STEP_MS), int(60000 / 80 / STEP_MS))
        bpm = 60000 / STEP_MS / lags[np.argmax(ac[lags])]
    show, beats, level = build(feat, nframes, bpm)

    out = os.path.join(a.outdir, "LightShow")
    os.makedirs(out, exist_ok=True)
    wav = os.path.join(out, a.name + ".wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.audio, "-ar", str(SR), "-ac", "2",
                    "-c:a", "pcm_s16le", "-map_metadata", "-1", wav], check=True)
    write_fseq(os.path.join(out, a.name + ".fseq"), show.d, a.name + ".wav")

    print(f"BPM {bpm:.1f}, {len(beats)} beats, {nframes} frames ({nframes * STEP_MS / 1000:.1f} s)")
    print("Bar levels:", "".join(str(v) for v in level))
    print("Closure commands:", {k: v for k, v in sorted(show.closure_cmds.items())})
    lit = (show.d[:, :46] > 0).mean(axis=0)
    print("Busiest channels:", sorted(range(1, 47), key=lambda c: -lit[c - 1])[:8])
    print("Written to", out)


if __name__ == "__main__":
    sys.exit(main())
