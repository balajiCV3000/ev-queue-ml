# EV Charging Queue Optimizer

A simulation platform for optimizing electric vehicle charging station assignments based on location, wait times, and energy needs.

![EV Charging Queue Optimizer](img/image.png)

---

## ⚠️ Patent Notice

**The core optimization methodology and algorithms in this project are covered by a published patent application (pending grant).** This code is provided for **educational and research purposes only**. Commercial use, redistribution, or implementation of the optimization algorithm — in whole or in part — requires explicit written permission. See the [License](#license) section for full terms.

---

## Overview

This project provides a web-based simulation environment to analyze and optimize electric vehicle charging station assignments. The system reduces wait times and improves charging infrastructure utilization through intelligent routing and queue management.

## Core Optimization Algorithm

*Implemented in `models/optimization.py` — patent pending.*

The algorithm evaluates multiple factors to assign EVs to optimal charging stations:

1. **Accessibility** — Determines if an EV can reach a station with current battery
2. **Route Proximity** — Prioritizes stations close to the EV's planned route
3. **Queue Length** — Considers current waiting times at each station
4. **Charging Time** — Calculates time needed based on current SoC and energy demand
5. **Total Time Cost** — Combines travel, wait, and charge time for overall optimization

## Features

- Real-time simulation of EV movements and charging needs
- Smart charging station assignment based on:
  - Battery level
  - Distance to charging stations
  - Station queue lengths
  - Estimated charging times
  - Route optimization
- Interactive map visualization
- Performance metrics tracking
- Customizable simulation parameters

## Technology Stack

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript, Chart.js
- **Maps API:** Google Maps Platform
- **Data Processing:** NumPy

## Project Structure

```
├── app.py                  # Main Flask application
├── config.py               # Configuration settings
├── models/
│   ├── ev.py               # Electric vehicle model
│   ├── station.py          # Charging station model
│   ├── simulation.py       # Simulation engine (Strategy pattern)
│   ├── optimization.py     # Charging assignment algorithm (patent pending)
│   └── maps_service.py     # Google Maps integration
├── static/
│   ├── css/                # Stylesheets
│   └── js/                 # Client-side scripts
├── templates/
│   └── index.html          # Main UI template
└── utils/
    └── data_generator.py   # Synthetic data generation
```

## Installation

1. Clone the repository:
```bash
   git clone https://github.com/yourusername/ev-charging-queue-optimizer.git
   cd ev-charging-queue-optimizer
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Create a local environment file:
```bash
   cp env.example .env
```

4. Update `.env` values as needed:
   - `GOOGLE_MAPS_API_KEY` for route generation
   - `DEBUG` for local debugging (`false` by default)
   - `BOOTSTRAP_SIMULATION` (`true` for full startup, `false` for lightweight smoke tests)
   - `APP_ENV` and `REQUIRE_MAPS_API_KEY` for production safety checks

## Usage

1. Start the server:
```bash
   python app.py
```

2. Open `http://127.0.0.1:5000` in your browser.

### Simulation Controls

- **Start / Stop** — Run or pause the simulation
- **Reset** — Return to initial state
- **Speed** — Adjust playback (1x–10x)
- **Generate New Data** — Create a new scenario with custom parameters:
  - Number of EVs
  - Number of charging stations
  - Number of geographic nodes
  - Number of routes

## License

This project is licensed under a custom license — see the [LICENSE](LICENSE) file for details.

The core optimization methodology is subject to a **pending patent**. Code is provided for educational and research purposes only. Commercial use or implementation of the optimization algorithm requires explicit written permission.

## Acknowledgments

- Google Maps Platform — geospatial services
- Chart.js — visualization components
- Flask — web framework

## Contact

For inquiries regarding licensing or commercial use: cvbalaji19672004@gmail.com
