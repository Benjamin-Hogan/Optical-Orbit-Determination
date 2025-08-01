#!/usr/bin/env python
"""Simulate topocentric RA/Dec observations from a TLE."""

from __future__ import annotations

import csv
import argparse
from typing import List

import numpy as np
from astropy.coordinates import (
    EarthLocation,
    AltAz,
    TEME,
    ITRS,
    ICRS,
)
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from astropy.time import Time
import astropy.units as u
from sgp4.api import Satrec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tle-line1", required=True, help="First line of the TLE")
    p.add_argument("--tle-line2", required=True, help="Second line of the TLE")
    p.add_argument("--lat", type=float, required=True, help="Observer latitude in degrees")
    p.add_argument("--lon", type=float, required=True, help="Observer longitude in degrees")
    p.add_argument("--alt", type=float, default=0.0, help="Observer altitude in meters")
    p.add_argument("--start", required=True, help="Start time UTC (ISO format)")
    p.add_argument("--minutes", type=float, default=0.0, help="Duration in minutes")
    p.add_argument("--step", type=float, default=60.0, help="Step in seconds between observations")
    p.add_argument("--output", default="observations.csv", help="Output CSV file")
    return p.parse_args()


def generate_times(start: Time, minutes: float, step: float) -> List[Time]:
    n = int(minutes * 60 / step) + 1
    return [start + i * step * u.s for i in range(n)]


def teme_to_topocentric(tle1: str, tle2: str, times: List[Time], location: EarthLocation):
    sat = Satrec.twoline2rv(tle1, tle2)
    rows = []
    for t in times:
        e, r, v = sat.sgp4(t.jd1, t.jd2)
        if e != 0:
            raise RuntimeError(f"SGP4 error {e}")
        r = np.array(r) * u.km
        v = np.array(v) * u.km / u.s
        teme = TEME(CartesianRepresentation(r, differentials=CartesianDifferential(v)), obstime=t)
        itrs = teme.transform_to(ITRS(obstime=t))
        altaz = itrs.transform_to(AltAz(obstime=t, location=location))
        icrs = altaz.transform_to(ICRS())
        rows.append({
            'time': t.utc.iso,
            'ra_deg': icrs.ra.deg,
            'dec_deg': icrs.dec.deg
        })
    return rows


def main() -> None:
    args = parse_args()
    start = Time(args.start)
    times = generate_times(start, args.minutes, args.step)
    location = EarthLocation(lat=args.lat*u.deg, lon=args.lon*u.deg, height=args.alt*u.m)
    rows = teme_to_topocentric(args.tle_line1, args.tle_line2, times, location)
    with open(args.output, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["time", "ra_deg", "dec_deg"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} observations to {args.output}")


if __name__ == "__main__":
    main()
