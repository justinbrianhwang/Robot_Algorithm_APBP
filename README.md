# APBP (Artificial Potential-Based Planner) PyBullet Demos

A small collection of PyBullet demonstration scripts that illustrate artificial-potential-based planning
behaviors: local minima / escape, rigid-body parking (3-point docking), obstacle-rich arena avoidance,
and a UAV takeoff + 3D visualization variant.

## Included files
- `apbp_pybullet_demo_local_minimum.py` — local-minimum prone APBP demo with optional tangential escape. (See file for controls.)

  <img width="622" height="296" alt="image" src="https://github.com/user-attachments/assets/3630a0ae-5263-42f0-b4e0-857fdd783758" />

- `apbp_parking_3points.py` — rigid 3-point (triangle) parking / docking using APBP and Kabsch-2D snap.

  <img width="641" height="296" alt="image" src="https://github.com/user-attachments/assets/238e99eb-a255-4113-b326-98c242565109" />


- `avoid_shit.py` — Arena + spawned spherical obstacles; proximity-weighted APBP avoidance with simple local-minimum rotation subplanner.

  <img width="664" height="368" alt="image" src="https://github.com/user-attachments/assets/014afaa4-19d5-44fd-b131-8ca6921496e3" />


- `apbp_uav_3d_triangle.py` — 3-point parking demo adapted for a UAV: starts on the ground and ascends to a triangular goal in the air.

<img width="749" height="425" alt="image" src="https://github.com/user-attachments/assets/cffc684e-c4fd-4dd0-b41e-d00fe6286be0" />

  

## Quick start

Requirements (recommended):
- Python 3.9+
- `pybullet`, `numpy`
- Optional: `imageio` (for GIF export in some scripts)

Install:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements_apbp.txt
```

## Usage examples (GUI / headless)
Each script supports a GUI run; a subset also supports headless GIF export or parameters via CLI.

- Local-minimum demo (GUI):
  ```bash
  python apbp_pybullet_demo_local_minimum.py --gui
  ```
  Headless (writes `apbp_demo.gif` if imageio is available):
  ```bash
  python apbp_pybullet_demo_local_minimum.py --gif
  ```
  Controls: Close window [X] to exit, `r`/`R` to reset, `Space` to pause/resume, `[`/`]` to slow/fast. fileciteturn1file0

- Rigid 3-point parking (GUI):
  ```bash
  python apbp_parking_3points.py
  ```
  Controls: Close [X], `r`/`R` reset, `Space` pause/resume, `[`/`]` speed control. This demo uses a Kabsch-2D snap to remove small docking errors. fileciteturn1file1

- Arena avoidance (`avoid_shit.py`):
  ```bash
  python avoid_shit.py
  ```
  Controls: `Space` pause/resume, `[`/`]` change simulation speed, `+`/`-` (or `=`/`_`) to increase/decrease spawn rate, `r`/`R` restart. The script spawns spherical obstacles that fall in and can be removed on contact. fileciteturn1file2

- UAV 3D triangle (GUI):
  ```bash
  python apbp_uav_3d_triangle.py
  ```
  This variant visualizes the 2D parking progress while the UAV ascends from ground to a target altitude and docks in the air. Controls are the same as the 3-point parking demo. fileciteturn1file3

## Notes & tips
- Scripts are intentionally conservative with control modes and rendering so they run on machines without GPU-accelerated OpenGL (e.g., llvmpipe) — if you experience low FPS, try headless mode (DIRECT) where available.
- `imageio` is used by `apbp_pybullet_demo_local_minimum.py` for optional GIF output; if you don't need GIFs you can omit installing it.
- All scripts contain in-file comments that describe the planner parameters, keyboard mappings, and tuning knobs — check the top of each file for detail. fileciteturn1file0

## Repository layout (suggested)
```
.
├── apbp_pybullet_demo_local_minimum.py
├── apbp_parking_3points.py
├── apbp_uav_3d_triangle.py
├── avoid_shit.py
├── README_APBP.md
└── requirements_apbp.txt
```

## License
Add a LICENSE file if you plan to publish — MIT is a permissive default recommendation.

--
