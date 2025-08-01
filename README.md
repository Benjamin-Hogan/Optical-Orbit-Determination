# Optical Orbit Determination

This repository provides a minimal pipeline to simulate optical observations of Earth satellites and determine their orbits using the Gauss method.

## Quick Start

1. Install Python dependencies (requires internet):
   ```bash
   pip install -r requirements.txt
   ```

2. Simulate observations from a TLE:
   ```bash
   python simulate_observations.py \
       --tle-line1 "1 25544U 98067A   24122.54791667  .00007428  00000-0  14123-3 0  9991" \
       --tle-line2 "2 25544  51.6414  23.4015 0002868  98.6301 261.5404 15.49816239396964" \
       --lat 33.35 --lon -111.79 --alt 0 \
       --start "2024-05-01T05:00:00" --minutes 2 --step 60 \
       --output obs.csv
   ```

3. Solve for the orbit using the Gauss method:
   ```bash
   python gauss_od.py obs.csv
   ```

The `obs.csv` file produced in step 2 contains simulated right ascension, declination and time values. The solver reads this file and prints the recovered orbital elements.

The code is intentionally simple and meant for experimentation with orbit determination techniques.
