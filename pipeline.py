#!/usr/bin/env python
"""Run the full optical orbit determination pipeline.

This script simulates observations from a TLE, recovers the orbit using the
Gauss method and then launches the interactive dashboard to visualize the
results.
"""

from __future__ import annotations

import argparse
import csv

from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u
from dash import Dash, dcc, html

import simulate_observations
import orbit_dashboard


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
    p.add_argument("--output", default="obs.csv", help="Output CSV file for the observations")
    return p.parse_args()


def simulate(args: argparse.Namespace) -> list[dict]:
    start = Time(args.start)
    times = simulate_observations.generate_times(start, args.minutes, args.step)
    location = EarthLocation(lat=args.lat * u.deg, lon=args.lon * u.deg, height=args.alt * u.m)
    rows = simulate_observations.teme_to_topocentric(args.tle_line1, args.tle_line2, times, location)
    with open(args.output, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["time", "ra_deg", "dec_deg"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} observations to {args.output}")
    return rows


def visualize(args: argparse.Namespace) -> None:
    ns = argparse.Namespace(csv_file=args.output, lat=args.lat, lon=args.lon, alt=args.alt)
    orbit, obs = orbit_dashboard.recover_orbit(ns)
    print(orbit.classical())

    ts, rs = orbit_dashboard.orbit_trajectory(orbit)
    fig_orbit = orbit_dashboard.make_orbit_figure(rs)
    fig_obs = orbit_dashboard.make_observation_figure(obs)
    fig_alt = orbit_dashboard.make_altitude_figure(ts, rs)

    app = Dash(__name__)
    app.layout = html.Div([
        html.H1("Orbit Dashboard"),
        html.Div([
            dcc.Graph(figure=fig_orbit),
            dcc.Graph(figure=fig_obs),
            dcc.Graph(figure=fig_alt),
        ], style={"display": "flex", "flexWrap": "wrap"}),
    ])
    app.run_server(debug=True)


def main() -> None:
    args = parse_args()
    simulate(args)
    visualize(args)


if __name__ == "__main__":
    main()
