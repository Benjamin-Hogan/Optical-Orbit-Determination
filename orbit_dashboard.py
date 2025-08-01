#!/usr/bin/env python
"""Interactive dashboard to visualize recovered orbits."""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
import plotly.graph_objects as go
from dash import Dash, dcc, html

from gauss_od import read_observations, los_vectors, site_vectors, gauss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv_file", help="CSV file with time,ra_deg,dec_deg columns")
    p.add_argument("--lat", type=float, required=True, help="Observer latitude in degrees")
    p.add_argument("--lon", type=float, required=True, help="Observer longitude in degrees")
    p.add_argument("--alt", type=float, default=0.0, help="Observer altitude in meters")
    return p.parse_args()


def recover_orbit(args: argparse.Namespace) -> tuple[Orbit, list[dict]]:
    obs = read_observations(args.csv_file)
    if len(obs) < 3:
        raise SystemExit("Need at least 3 observations")
    obs = obs[:3]
    times = [Time(o["time"]) for o in obs]
    location = EarthLocation(lat=args.lat * u.deg, lon=args.lon * u.deg, height=args.alt * u.m)
    rho_hats = los_vectors(obs)
    Rs = site_vectors(times, location)
    r_vec, v_vec = gauss(rho_hats, Rs, times)
    orbit = Orbit.from_vectors(Earth, r_vec * u.km, v_vec * u.km / u.s, epoch=times[1])
    return orbit, obs


def orbit_trajectory(orbit: Orbit, num: int = 200):
    period = orbit.period.to(u.s).value
    ts = np.linspace(0, period, num)
    rs = np.array([orbit.propagate(t * u.s).r.to(u.km).value for t in ts])
    return ts, rs


def make_orbit_figure(rs: np.ndarray) -> go.Figure:
    xs, ys, zs = rs.T
    r_earth = Earth.R.to(u.km).value
    theta = np.linspace(0, 2 * np.pi, 60)
    phi = np.linspace(0, np.pi, 30)
    xe = r_earth * np.outer(np.cos(theta), np.sin(phi))
    ye = r_earth * np.outer(np.sin(theta), np.sin(phi))
    ze = r_earth * np.outer(np.ones_like(theta), np.cos(phi))

    fig = go.Figure()
    fig.add_trace(go.Surface(x=xe, y=ye, z=ze, colorscale="Blues", showscale=False, opacity=0.5, name="Earth"))
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name="Orbit"))
    fig.update_layout(scene_aspectmode="data", scene=dict(xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)"))
    return fig


def make_observation_figure(obs: list[dict]) -> go.Figure:
    times = [datetime.fromisoformat(o["time"]) for o in obs]
    ra = [float(o["ra_deg"]) for o in obs]
    dec = [float(o["dec_deg"]) for o in obs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=ra, mode="markers+lines", name="RA (deg)"))
    fig.add_trace(go.Scatter(x=times, y=dec, mode="markers+lines", name="Dec (deg)"))
    fig.update_layout(xaxis_title="Time", yaxis_title="Degrees")
    return fig


def make_altitude_figure(ts: np.ndarray, rs: np.ndarray) -> go.Figure:
    r_earth = Earth.R.to(u.km).value
    alt = np.linalg.norm(rs, axis=1) - r_earth
    fig = go.Figure(go.Scatter(x=ts / 60.0, y=alt))
    fig.update_layout(xaxis_title="Minutes", yaxis_title="Altitude (km)")
    return fig


def main() -> None:
    args = parse_args()
    orbit, obs = recover_orbit(args)
    ts, rs = orbit_trajectory(orbit)

    fig_orbit = make_orbit_figure(rs)
    fig_obs = make_observation_figure(obs)
    fig_alt = make_altitude_figure(ts, rs)

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


if __name__ == "__main__":
    main()

