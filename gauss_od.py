#!/usr/bin/env python
"""Recover an orbit from angles-only observations using the Gauss method."""

from __future__ import annotations

import csv
import argparse
from typing import List

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from poliastro.twobody import Orbit
from poliastro.bodies import Earth

MU_EARTH = Earth.k.to(u.km**3 / u.s**2).value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv_file", help="CSV file with time,ra_deg,dec_deg columns")
    p.add_argument("--lat", type=float, required=True, help="Observer latitude in degrees")
    p.add_argument("--lon", type=float, required=True, help="Observer longitude in degrees")
    p.add_argument("--alt", type=float, default=0.0, help="Observer altitude in meters")
    return p.parse_args()


def read_observations(fname: str) -> List[dict]:
    with open(fname) as f:
        reader = csv.DictReader(f)
        return list(reader)


def los_vectors(obs: List[dict]):
    return [SkyCoord(float(o["ra_deg"])*u.deg, float(o["dec_deg"])*u.deg, frame="icrs").cartesian.xyz.value for o in obs]


def site_vectors(obs_times: List[Time], loc: EarthLocation):
    return [loc.get_gcrs(obstime=t).cartesian.xyz.to(u.km).value for t in obs_times]


def gauss(rho_hats, Rs, times):
    t1, t2, t3 = times
    tau1 = (t1 - t2).to(u.s).value
    tau3 = (t3 - t2).to(u.s).value
    tau = tau3 - tau1

    rho1_hat, rho2_hat, rho3_hat = rho_hats
    R1, R2, R3 = Rs

    D0 = np.dot(rho1_hat, np.cross(rho2_hat, rho3_hat))
    D11 = np.dot(np.cross(R1, rho2_hat), rho3_hat)
    D12 = np.dot(np.cross(R2, rho2_hat), rho3_hat)
    D13 = np.dot(np.cross(R3, rho2_hat), rho3_hat)
    D21 = np.dot(np.cross(R1, rho1_hat), rho3_hat)
    D22 = np.dot(np.cross(R2, rho1_hat), rho3_hat)
    D23 = np.dot(np.cross(R3, rho1_hat), rho3_hat)
    D31 = np.dot(np.cross(R1, rho1_hat), rho2_hat)
    D32 = np.dot(np.cross(R2, rho1_hat), rho2_hat)
    D33 = np.dot(np.cross(R3, rho1_hat), rho2_hat)

    A1 = tau3 / tau
    A3 = -tau1 / tau
    B1 = A1 / 6.0 * (tau**2 - tau3**2)
    B3 = A3 / 6.0 * (tau**2 - tau1**2)

    A = (A1 * D21 - D22 + A3 * D23) / (-D0)
    B = (B1 * D21 + B3 * D23) / (-D0)

    E = np.dot(R2, rho2_hat)
    R2_norm = np.linalg.norm(R2)

    a = -(A**2 + 2*A*E + R2_norm**2)
    b = -2 * MU_EARTH * B * (A + E)
    c = -MU_EARTH**2 * B**2

    coeffs = [1, 0, a, 0, 0, b, 0, 0, c]
    roots = np.roots(coeffs)
    r2_mag = np.real(roots[np.isreal(roots) & (roots > 0)][0])

    f1 = 1 - 0.5 * MU_EARTH * tau1**2 / r2_mag**3
    f3 = 1 - 0.5 * MU_EARTH * tau3**2 / r2_mag**3
    g1 = tau1 - MU_EARTH * tau1**3 / (6 * r2_mag**3)
    g3 = tau3 - MU_EARTH * tau3**3 / (6 * r2_mag**3)

    c1 = g3 / (f1 * g3 - f3 * g1)
    c3 = -g1 / (f1 * g3 - f3 * g1)
    c2 = -c1 - c3

    rho1 = (c1 * D11 + c2 * D21 + c3 * D31) / (c1 * D0)
    rho2 = (c1 * D12 + c2 * D22 + c3 * D32) / (c2 * D0)
    rho3 = (c1 * D13 + c2 * D23 + c3 * D33) / (c3 * D0)

    r1 = rho1 * rho1_hat - R1
    r2 = rho2 * rho2_hat - R2
    r3 = rho3 * rho3_hat - R3

    v2 = (f3 * r1 - f1 * r3) / (f1 * g3 - f3 * g1)
    return r2, v2


def main() -> None:
    args = parse_args()
    obs = read_observations(args.csv_file)
    if len(obs) < 3:
        raise SystemExit("Need at least 3 observations")
    obs = obs[:3]
    times = [Time(o['time']) for o in obs]
    location = EarthLocation(lat=args.lat*u.deg, lon=args.lon*u.deg, height=args.alt*u.m)
    rho_hats = los_vectors(obs)
    Rs = site_vectors(times, location)
    r_vec, v_vec = gauss(rho_hats, Rs, times)
    orbit = Orbit.from_vectors(Earth, r_vec * u.km, v_vec * u.km / u.s, epoch=times[1])
    print(orbit.classical())


if __name__ == "__main__":
    main()
