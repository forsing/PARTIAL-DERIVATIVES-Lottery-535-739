import csv
import math
from pathlib import Path

CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7_4688_k75.csv")
# CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7_4688_k75_loto_2966.csv")
# CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7_4688_k75_loto_plus_1722.csv")

SEED = 39
LR = 0.1
STEPS = 50
WARMUP = 1
N_MAIN = 7
N_MAX = 39


def validate_row(row, line):
    if (
        len(row) != N_MAIN
        or len(set(row)) != N_MAIN
        or any(not 1 <= x <= N_MAX for x in row)
    ):
        raise ValueError(
            f"Neispravan red {line}: {row}. "
            "Potrebno je 7 različitih brojeva 1-39."
        )


def load_data(path):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        for line, fields in enumerate(csv.reader(file), 1):
            if not fields:
                continue
            try:
                row = tuple(int(x) for x in fields)
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: očekuju se celi brojevi."
                ) from error
            validate_row(row, line)
            rows.append(row)
    if not rows:
        raise ValueError("CSV je prazan.")
    return rows


def features(history, maximum):
    t = len(history)
    freq = [0] * (maximum + 1)
    last = [0] * (maximum + 1)
    for i, draw in enumerate(history, 1):
        for num in draw:
            freq[num] += 1
            last[num] = i
    out = [None] * (maximum + 1)
    for num in range(1, maximum + 1):
        f = freq[num] / t
        gap = (t - last[num]) / t if last[num] else 1.0
        out[num] = (f, gap)
    return out


def sigmoid(z):
    if z >= 20:
        return 1.0
    if z <= -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def train(draws, maximum, steps=STEPS, lr=LR):
    # bias, težina učestalosti, težina razmaka — minus-gradijent za težine
    w = [0.0, SEED / 39.0, SEED / 39.0]
    n = len(draws)
    for _ in range(steps):
        g = [0.0, 0.0, 0.0]
        count = 0
        freq = [0] * (maximum + 1)
        last = [0] * (maximum + 1)
        for t in range(n - 1):
            for num in draws[t]:
                freq[num] += 1
                last[num] = t + 1
            if t + 1 < WARMUP:
                continue
            hist_len = t + 1
            drawn = set(draws[t + 1])
            for num in range(1, maximum + 1):
                f = freq[num] / hist_len
                gap = (hist_len - last[num]) / hist_len if last[num] else 1.0
                p = sigmoid(w[0] + w[1] * f + w[2] * gap)
                err = p - (1.0 if num in drawn else 0.0)
                g[0] += err
                g[1] += err * f
                g[2] += err * gap
                count += 1
        inv = 1.0 / count
        w[0] -= lr * g[0] * inv
        w[1] -= lr * g[1] * inv
        w[2] -= lr * g[2] * inv
    return w


def rank(history, w, maximum, k):
    feats = features(history, maximum)
    scored = []
    for num in range(1, maximum + 1):
        f, gap = feats[num]
        s = w[0] + w[1] * f + w[2] * gap
        scored.append((-s, num))
    scored.sort()
    return tuple(sorted(n for _, n in scored[:k]))


def freq_rank(history, maximum, k):
    counts = [0] * (maximum + 1)
    for draw in history:
        for num in draw:
            counts[num] += 1
    scored = [(-counts[num], num) for num in range(1, maximum + 1)]
    scored.sort()
    return tuple(sorted(n for _, n in scored[:k]))


def mean_hits(draws, start, w, maximum, k, use_freq):
    hits = []
    for t in range(start, len(draws)):
        pred = (
            freq_rank(draws[:t], maximum, k)
            if use_freq
            else rank(draws[:t], w, maximum, k)
        )
        hits.append(len(set(pred) & set(draws[t])))
    return sum(hits) / len(hits) if hits else 0.0


def run_pool(draws, maximum, k, title):
    n = len(draws)
    start = int(n * 0.80)
    w_hold = train(draws[:start], maximum)
    gd = mean_hits(draws, start, w_hold, maximum, k, False)
    fr = mean_hits(draws, start, w_hold, maximum, k, True)
    w = train(draws, maximum)
    nxt = rank(draws, w, maximum, k)
    print(title)
    print(f"  težine bias,freq,gap: {w[0]:.6f}, {w[1]:.6f}, {w[2]:.6f}")
    print(f"  test gradijent: {gd:.4f}")
    print(f"  test učestalost: {fr:.4f}")
    return nxt


def main():
    rows = load_data(CSV_PATH)
    print(f"CSV: {CSV_PATH}")
    print(f"Učitano redova: {len(rows)}")
    print(f"SEED={SEED} (konstanta, bez RNG), LR={LR}, STEPS={STEPS}")

    nxt = run_pool(rows, N_MAX, N_MAIN, "Brojevi 7/39")

    print("\nnext")
    print(",".join(map(str, nxt)))


if __name__ == "__main__":
    main()



"""
CSV: /Users/4c/Desktop/GHQ/data/loto7_4688_k75.csv
Učitano redova: 4688
SEED=39 (konstanta, bez RNG), LR=0.1, STEPS=50
Brojevi 7/39
  težine bias,freq,gap: -1.054393, 0.810690, 0.990710
  test gradijent: 1.3060
  test učestalost: 1.2889

next
8,11,23,26,32,34,37
"""



"""
CSV: /Users/4c/Desktop/GHQ/data/loto7_4688_k75_loto_2966.csv
Učitano redova: 2966
SEED=39 (konstanta, bez RNG), LR=0.1, STEPS=50
Brojevi 7/39
  težine bias,freq,gap: -1.056710, 0.810245, 0.986111
  test gradijent: 1.2694
  test učestalost: 1.2104

next
8,21,22,23,26,33,35
"""



"""
CSV: /Users/4c/Desktop/GHQ/data/loto7_4688_k75_loto_plus_1722.csv
Učitano redova: 1722
SEED=39 (konstanta, bez RNG), LR=0.1, STEPS=50
Brojevi 7/39
  težine bias,freq,gap: -1.061405, 0.809720, 0.975255
  test gradijent: 1.2348
  test učestalost: 1.2029

next
2,11,23,26,32,34,37
"""




"""
+---------------------------------------------------------------+
|  PARCIJALNI IZVODI                           |
+---------------------------------------------------------------+
|  ∂f/∂x        mrdni x, zamrzni ostalo, diferenciraj           |
|  frozen term  nema x  ->  izvod je 0                          |
|  frozen coef  množi x-deo  ->  ide uz njega                   |
|  pretend y=7  mentalni trik da zamrzavanje deluje stvarno     |
|  ∇f           svi parcijalni složeni, najstrmije uzbrdo       |
|  -∇f          najstrmije nizbrdo, smer koj AI trenira         |
|  fxy = fyx    mešoviti parcijalni se poklapaju, redosled slobodan |
|  chain rule   pomnoži duž svake staze, saber staze            |
|  Δf ≈ Σ (∂f/∂xᵢ)·Δxᵢ    formula računa                        |
+---------------------------------------------------------------+
"""



"""
Tehnika gradijentni spust može da posluži za treniranje loto modela.
Za 5/35 + dopunski 1/10:
- Učitavanje CSV-a.
- Računanje učestalosti i razmaka između pojavljivanja, koristeći samo prethodna izvlačenja.
- Učenje težina tih pokazatelja gradijentnim spustom.
- Odvojeno ocenjivanje 35 glavnih i 10 dopunskih brojeva.
- Provera na kasnijim izvlačenjima i poređenje sa prethodnim modelom učestalosti.
Smisao primene je da model nauči težine iz podataka. 
Da li to poboljšava predikciju mora da pokaže test; sami izvodi ne daju prednost u loto izvlačenju.
"""



"""

istorija kola + funkcija greške + isti minus-gradijent za težine

CSV, učestalost i razmaci samo iz prethodnih kola, 
učenje težina spustom, odvojeno 35 glavnih i 10 dopunskih, pa provera na kasnijim kolima. 
Test odlučuje da li vredi; sami izvodi ne. 

Gradijentni spust radi sa bilo kojom funkcijom greške. 
Za loto to znači: brojevi iz istorije → pokazatelji (npr. učestalost, razmak) → težine → greška na sledećem kolu → ažuriranje težina. 

Da li će predikcija biti bolja od učestalosti vidi se tek na testu, ne iz same formule.

Isti spust kao, učestalost i razmak iz prethodnih kola, odvojene težine za 5/35 i dopunski 1/10, jedan next.

CSV
učestalost i razmak samo iz prethodnih kola
težine gradijentnim spustom
odvojeno 5/35 i dopunski 1/10
provera na kasnijim kolima vs učestalost

next: 7,9,26,27,30,9

Na testu: glavni 0.9474 vs 0.6842; dopunski 0.0789 vs 0.0526.




— isti spust (učestalost + razmak), 7/39, CSV, next.

next: 8,11,23,26,32,34,37

Test: gradijent 1.3060, učestalost 1.2889. 

"""
