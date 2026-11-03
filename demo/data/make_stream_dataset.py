"""Generate demo/data/room_occupancy_stream.csv (synthetic, reproducible).

A simulated smart-building sensor stream: one row every 10 minutes for
8000 readings (~55 days). The task is to classify how busy a meeting room is:

    occupancy = 0 empty, 1 a few people, 2 busy

Features
    hour, temperature (C), humidity (%), light (lux), co2 (ppm),
    noise_db, day_type (text: weekday / weekend / holiday)

Realistic streaming problems are built in on purpose:
    * ~3% missing sensor values (empty cells),
    * a holiday category that first appears late in the stream,
    * concept drift at row 5000: the light sensor is replaced and now reads
      ~40% lower, so a tree trained on old data misreads the light feature
      until it retrains on new data.

Only NumPy is used. Run:  python demo/data/make_stream_dataset.py
"""

from pathlib import Path

import numpy as np

N, DRIFT_AT, SEED = 8000, 5000, 2026


def generate(n=N, drift_at=DRIFT_AT, seed=SEED):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    hour = (t * 10 // 60) % 24
    day = t * 10 // (60 * 24)
    day_type = np.where(day % 7 >= 5, "weekend", "weekday").astype(object)
    holiday_days = rng.choice(np.arange(40, day.max() + 1), size=3, replace=False)
    day_type[np.isin(day, holiday_days)] = "holiday"

    work = (day_type == "weekday") & (hour >= 8) & (hour <= 18)
    lunch = work & (hour >= 12) & (hour <= 13)
    # latent number of people in the room
    people = np.where(work, rng.poisson(6, n), rng.poisson(0.3, n))
    people = np.where(lunch, rng.poisson(1.5, n), people)
    occupancy = np.digitize(people, [1, 6])               # 0, 1-5, 6+

    daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    light = 40 + 300 * daylight + 45 * people + rng.normal(0, 40, n)
    light[t >= drift_at] *= 0.6                           # new, dimmer sensor
    co2 = 420 + 55 * people + rng.normal(0, 60, n)
    temperature = 20.5 + 0.25 * people + 1.5 * daylight + rng.normal(0, 0.6, n)
    humidity = 40 + 0.8 * people + rng.normal(0, 4, n)
    noise = 32 + 4.0 * np.sqrt(people) + rng.normal(0, 3, n)

    num = np.column_stack([hour, temperature, humidity, light, co2, noise]).astype(float)
    num[:, 1:] = np.round(num[:, 1:], 1)
    num[rng.random(num.shape) < 0.03] = np.nan
    num[:, 0] = hour                                       # the clock never fails
    return num, day_type, occupancy


def main(path=Path(__file__).with_name("room_occupancy_stream.csv")):
    num, day_type, occ = generate()
    header = "hour,temperature,humidity,light,co2,noise_db,day_type,occupancy"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for row, d, y in zip(num, day_type, occ):
            cells = ["" if np.isnan(v) else (str(int(v)) if j == 0 else f"{v:.1f}")
                     for j, v in enumerate(row)]
            fh.write(",".join(cells + [d, str(y)]) + "\n")
    print(f"wrote {path} ({len(occ)} rows), class counts {np.bincount(occ)}")


if __name__ == "__main__":
    main()
