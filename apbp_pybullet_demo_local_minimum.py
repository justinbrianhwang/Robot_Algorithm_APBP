# -*- coding: utf-8 -*-
"""
APBP (Artificial Potential-Based Planner) demo in PyBullet
- 2D (x-y plane) point robot with attractive + repulsive fields
- 3 obstacles arranged to create a Local Minimum (LM) first,
  then escape using a tangential push (when enabled).

Controls (GUI mode):
  - Close window [X] : terminate the whole program
  - r or R           : reset and restart the episode
  - Space            : pause/resume (toggle)
  - [ / ]            : slow down / speed up (0.1x ~ 8x, step 1.25x)
HUD:
  - Hidden by default. Use --hud to show on-screen text.
Headless:
  - One episode runs and (optionally) writes apbp_demo.gif, then exits
"""

import math
import argparse
import time

# ------------------------ imports ------------------------
try:
    import pybullet as p
    import pybullet_data
except Exception as e:
    raise SystemExit(
        "This script requires 'pybullet'. Install with:\n"
        "    pip install pybullet\n"
        f"Import error: {e}"
    )

try:
    import imageio.v2 as imageio
    HAVE_IMAGEIO = True
except Exception:
    HAVE_IMAGEIO = False

# Keyboard flags (version-safe)
KEY_WAS_TRIGGERED = getattr(p, "KEY_WAS_TRIGGERED", 1 << 0)
KEY_IS_DOWN       = getattr(p, "KEY_IS_DOWN",       1 << 1)
KEY_SPACE         = getattr(p, "B3G_SPACE", 32)  # fallback to ASCII space
KEY_LBRACKET      = ord('[')
KEY_RBRACKET      = ord(']')

# ------------------------ vector utils ------------------------
def v_add(a, b): return (a[0]+b[0], a[1]+b[1])
def v_sub(a, b): return (a[0]-b[0], a[1]-b[1])
def v_scale(a, s): return (a[0]*s, a[1]*s)
def v_len(a): return math.hypot(a[0], a[1])
def v_norm(a, eps=1e-12):
    L = v_len(a);  return (0.0, 0.0) if L < eps else (a[0]/L, a[1]/L)
def rot90(a, sign=1):  # (x,y)->(-y,x) or (y,-x)
    return (-sign*a[1], sign*a[0])

# ------------------------ planner ------------------------
class APFPlanner:
    """
    Attractive + repulsive potentials with optional tangential escape when stuck.
    Tuned to create a Local Minimum first.
    """
    def __init__(self, goal_xy, obstacles, zeta=0.8, d_star=1.2,
                 eta=220.0, Q_star=1.6, step_size=0.04,
                 stuck_steps=80, stuck_eps=1e-5,
                 force_eps=0.02, force_stuck_steps=35,
                 escape_gain=1.0, allow_escape=False):
        self.goal = goal_xy
        self.obs = obstacles
        self.zeta = zeta; self.d_star = d_star
        self.eta  = eta;  self.Q_star = Q_star
        self.step = step_size
        self.stuck_steps = stuck_steps; self.stuck_eps = stuck_eps
        self.force_eps = force_eps;     self.force_stuck_steps = force_stuck_steps
        self.escape_gain = escape_gain
        self.allow_escape = allow_escape

        self.best_U = float('inf')
        self.no_improve = 0
        self.weak_force_count = 0
        self.escape_dir = +1

    # --- potentials & forces ---
    def _att_pot_force(self, x):
        gx, gy = self.goal
        dx, dy = (x[0]-gx, x[1]-gy)
        d = math.hypot(dx, dy)
        if d <= self.d_star:
            U_att = 0.5 * self.zeta * d * d
            F = (-self.zeta * dx, -self.zeta * dy)
        else:
            U_att = self.zeta * self.d_star * d - 0.5 * self.zeta * self.d_star * self.d_star
            F = ((-self.zeta * self.d_star / d) * dx, (-self.zeta * self.d_star / d) * dy) if d > 1e-12 else (0.0, 0.0)
        return U_att, F

    def _rep_pot_force_one(self, x, c, R):
        eps = 1e-9
        dx, dy = (x[0]-c[0], x[1]-c[1])
        d_center = max(math.hypot(dx, dy), eps)
        rho = max(d_center - R, eps)  # distance to obstacle surface
        if rho > self.Q_star:
            return 0.0, (0.0, 0.0)
        U_rep = 0.5 * self.eta * (1.0/rho - 1.0/self.Q_star)**2
        coeff = self.eta * (1.0/rho - 1.0/self.Q_star) * (1.0/(rho*rho)) * (1.0/d_center)
        F = (coeff * dx, coeff * dy)
        return U_rep, F

    def pot_and_force(self, x):
        U_att, F_att = self._att_pot_force(x)
        Urep = 0.0; Fx, Fy = F_att
        for o in self.obs:
            u, f = self._rep_pot_force_one(x, o["c"], o["r"])
            Urep += u; Fx += f[0]; Fy += f[1]
        return U_att + Urep, (Fx, Fy)

    # --- LM detection & escape ---
    def _update_stuck_flags(self, U_now, F_norm):
        stuck = False
        if U_now < self.best_U - self.stuck_eps:
            self.best_U = U_now
            self.no_improve = 0
            self.weak_force_count = 0
        else:
            self.no_improve += 1
            if F_norm < self.force_eps:
                self.weak_force_count += 1
            else:
                self.weak_force_count = max(0, self.weak_force_count - 1)
            if self.no_improve > self.stuck_steps or self.weak_force_count > self.force_stuck_steps:
                stuck = True
        return stuck

    def _nearest_obstacle(self, x):
        best = None; best_val = float('inf')
        for o in self.obs:
            d = math.hypot(x[0]-o["c"][0], x[1]-o["c"][1]) - o["r"]
            if d < best_val: best_val = d; best = o
        return best, best_val

    def _choose_escape_dir(self, x, F):
        o, _ = self._nearest_obstacle(x)
        if o is None: return +1
        n = v_norm(v_sub(x, o["c"]))
        t1, t2 = rot90(n, +1), rot90(n, -1)
        U1, _ = self.pot_and_force(v_add(x, v_scale(t1, self.step)))
        U2, _ = self.pot_and_force(v_add(x, v_scale(t2, self.step)))
        return +1 if U1 < U2 else -1

    def next_step(self, x):
        U, F = self.pot_and_force(x)
        F_norm = v_len(F)
        stuck = self._update_stuck_flags(U, F_norm)

        mode = "gradient"
        dir_vec = F
        if stuck and self.allow_escape:
            mode = "escape"
            o, _ = self._nearest_obstacle(x)
            if o is not None:
                n = v_norm(v_sub(x, o["c"]))
                sgn = self._choose_escape_dir(x, F)
                self.escape_dir = sgn
                t = rot90(n, sgn)
                dir_vec = v_add(F, v_scale(t, self.escape_gain))

        d = v_norm(dir_vec)
        x_next = v_add(x, v_scale(d, self.step))
        return x_next, U, mode, stuck, F

# ------------------------ world & camera ------------------------
def create_world(start_xy, goal_xy, obstacles):
    p.resetSimulation()
    p.setGravity(0, 0, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    plane = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane, -1, rgbaColor=[1,1,1,1])

    def add_marker(xy, color=(0,1,0,1), radius=0.08, z=0.06):
        vs = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        cs = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cs,
                                 baseVisualShapeIndex=vs,
                                 basePosition=(xy[0], xy[1], z))

    add_marker(start_xy, color=(0, 0.9, 0.3, 1), radius=0.09)
    add_marker(goal_xy,  color=(0.7, 0.3, 1.0, 1), radius=0.09)

    for o in obstacles:
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=o["r"])
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=o["r"],
                                  rgbaColor=o.get("color", (0.8, 0.1, 0.1, 0.95)))
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                          baseVisualShapeIndex=vis,
                          basePosition=(o["c"][0], o["c"][1], o["r"]))

    # Robot
    robot_r = 0.05
    robot_vis = p.createVisualShape(p.GEOM_SPHERE, radius=robot_r, rgbaColor=[0.1,0.1,0.1,1])
    robot_col = p.createCollisionShape(p.GEOM_SPHERE, radius=robot_r)
    robot_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=robot_col,
                                 baseVisualShapeIndex=robot_vis,
                                 basePosition=(start_xy[0], start_xy[1], robot_r))
    return robot_id

def make_view_matrix(eye, target, up=(0,0,1)):
    # version-safe wrapper for computeViewMatrix
    try:
        return p.computeViewMatrix(cameraEyePosition=eye,
                                   cameraTargetPosition=target,
                                   cameraUpVector=up)
    except TypeError:
        return p.computeViewMatrix(eye, target, up)

# ---------- small helpers for optional HUD ----------
def add_text_if(show_hud, text, pos, color, size, lifetime):
    if not show_hud:
        return None
    return p.addUserDebugText(text, pos, textColorRGB=color, textSize=size, lifeTime=lifetime)

def show_temp_notice(show_hud, msg, color, sec=1.0):
    if not show_hud: 
        print(msg)
        return
    add_text_if(True, msg, [-4.9, 3.0, 0.02], color, 1.2, sec)

# ------------------------ demo (headless) ------------------------
def run_headless_episode(start_xy, goal_xy, obstacles,
                         zeta, d_star, eta, Q_star,
                         step_size, hold_stuck_steps,
                         gif_name):
    robot_id = create_world(start_xy, goal_xy, obstacles)
    width, height = 980, 560
    viewMatrix = make_view_matrix([0, -7.2, 8.2], [0.6, 0.5, 0.0], (0,0,1))
    projMatrix = p.computeProjectionMatrixFOV(fov=60, aspect=width/height, nearVal=0.1, farVal=30)

    frames = []
    def capture():
        if not HAVE_IMAGEIO: return
        rgba = p.getCameraImage(width, height, viewMatrix, projMatrix)[2]
        frames.append(rgba)

    for a,b in [([-5,-4,0],[5,-4,0]), ([5,-4,0],[5,4,0]), ([5,4,0],[-5,4,0]), ([-5,4,0],[-5,-4,0]),
                ([-5,0,0],[5,0,0]), ([0,-4,0],[0,4,0])]:
        p.addUserDebugLine(a, b, [0.2,0.2,0.2], 1, 0)

    planner = APFPlanner(goal_xy, obstacles,
                         zeta=zeta, d_star=d_star, eta=eta, Q_star=Q_star,
                         step_size=step_size, escape_gain=1.0,
                         allow_escape=False,
                         stuck_steps=80, stuck_eps=1e-5,
                         force_eps=0.02, force_stuck_steps=35)

    x = start_xy; prev_xyz = (x[0], x[1], 0.05)
    hold = -1; tol_goal = 0.14
    for t in range(5000):
        x_next, U, mode, stuck, F = planner.next_step(x)
        xyz = (x_next[0], x_next[1], 0.05)
        p.resetBasePositionAndOrientation(robot_id, xyz, [0,0,0,1])
        p.addUserDebugLine(prev_xyz, xyz, [0,0,0], 2, 0)
        prev_xyz = xyz
        if (stuck and not planner.allow_escape and hold < 0):
            hold = hold_stuck_steps
        if hold >= 0:
            hold -= 1
            if hold == 0: planner.allow_escape = True
        capture()
        p.stepSimulation()
        x = x_next
        if v_len(v_sub(x, goal_xy)) < tol_goal:
            break

    if HAVE_IMAGEIO and frames:
        try:
            imageio.mimsave(gif_name, frames, fps=50)
            print(f"Wrote {gif_name} with {len(frames)} frames.")
        except Exception as e:
            print(f"GIF write failed: {e}")

# ------------------------ demo (GUI loop) ------------------------
def run_gui_loop(start_xy, goal_xy, obstacles,
                 zeta, d_star, eta, Q_star,
                 step_size, hold_stuck_steps,
                 show_hud=False):
    """
    GUI main loop: never auto-terminates.
    - Close the window [X] to exit the program.
    - Press r / R to reset the episode at any time.
    - Press Space to pause/resume at any time.
    - Press [ / ] to slow down / speed up.
    HUD text is shown only if show_hud=True.
    """
    SIM_HZ_BASE = 120.0
    SPEED_MIN, SPEED_MAX = 0.1, 8.0

    def init_episode():
        p.removeAllUserDebugItems()
        robot_id = create_world(start_xy, goal_xy, obstacles)
        p.resetDebugVisualizerCamera(cameraDistance=6.4, cameraYaw=30, cameraPitch=-40,
                                     cameraTargetPosition=[0.6, 0.5, 0.0])

        # frame & axis
        for a,b in [([-5,-4,0],[5,-4,0]), ([5,-4,0],[5,4,0]), ([5,4,0],[-5,4,0]), ([-5,4,0],[-5,-4,0])]:
            p.addUserDebugLine(a, b, [0,0,0], 3, 0)
        p.addUserDebugLine([-5,0,0],[5,0,0],[0.6,0.6,0.6],1,0)
        p.addUserDebugLine([0,-4,0],[0,4,0],[0.6,0.6,0.6],1,0)

        planner = APFPlanner(goal_xy, obstacles,
                             zeta=zeta, d_star=d_star, eta=eta, Q_star=Q_star,
                             step_size=step_size, escape_gain=1.0,
                             allow_escape=False,
                             stuck_steps=80, stuck_eps=1e-5,
                             force_eps=0.02, force_stuck_steps=35)
        state = {
            "robot_id": robot_id,
            "planner": planner,
            "x": start_xy,
            "prev_xyz": (start_xy[0], start_xy[1], 0.05),
            "hud_id": None,
            "hold": -1,
            "tol_goal": 0.14,
            "frozen": False,   # goal reached => freeze
            "paused": False,   # Space toggles this
            "speed": 1.0,      # [ / ] adjust this
        }
        return state

    s = init_episode()
    if show_hud:
        print("GUI started. [X]=quit, [r/R]=restart, [Space]=pause/resume, [ / ]=speed (HUD ON)")
    else:
        print("GUI started. [X]=quit, [r/R]=restart, [Space]=pause/resume, [ / ]=speed (HUD OFF)")

    while p.isConnected():
        # ---- keyboard ----
        keys = p.getKeyboardEvents()
        if keys:
            # restart
            if ((ord('r') in keys and (keys[ord('r')] & KEY_WAS_TRIGGERED)) or
                (ord('R') in keys and (keys[ord('R')] & KEY_WAS_TRIGGERED))):
                s = init_episode()
                show_temp_notice(show_hud, "Restarted", [0.1,0.4,0.9], 1.0)
                # debounce
                for _ in range(int(0.15*SIM_HZ_BASE)):
                    p.stepSimulation(); time.sleep(1/SIM_HZ_BASE)
                continue

            # pause/resume
            if (KEY_SPACE in keys) and (keys[KEY_SPACE] & KEY_WAS_TRIGGERED):
                s["paused"] = not s["paused"]
                show_temp_notice(show_hud,
                                 "PAUSED (Space to resume)" if s["paused"] else "RESUMED",
                                 [0.7,0.1,0.1] if s["paused"] else [0.1,0.5,0.1],
                                 1.0)

            # speed down/up with [ / ]
            if (KEY_LBRACKET in keys) and (keys[KEY_LBRACKET] & KEY_WAS_TRIGGERED):
                s["speed"] = max(SPEED_MIN, s["speed"] / 1.25)
                show_temp_notice(show_hud, f"Speed x{s['speed']:.2f}", [0.2,0.2,0.9], 1.0)
            if (KEY_RBRACKET in keys) and (keys[KEY_RBRACKET] & KEY_WAS_TRIGGERED):
                s["speed"] = min(SPEED_MAX, s["speed"] * 1.25)
                show_temp_notice(show_hud, f"Speed x{s['speed']:.2f}", [0.2,0.2,0.9], 1.0)

        # ---- simulation step ----
        if not s["frozen"] and not s["paused"]:
            x_next, U, mode, stuck, F = s["planner"].next_step(s["x"])
            xyz = (x_next[0], x_next[1], 0.05)
            p.resetBasePositionAndOrientation(s["robot_id"], xyz, [0,0,0,1])
            p.addUserDebugLine(s["prev_xyz"], xyz, [0,0,0], 2, 0)
            s["prev_xyz"] = xyz

            # tangent arrow in escape mode (line annotation only)
            if s["planner"].allow_escape and mode == "escape":
                o, _ = s["planner"]._nearest_obstacle(s["x"])
                if o is not None:
                    n = v_norm(v_sub(s["x"], o["c"]))
                    tdir = rot90(n, s["planner"].escape_dir)
                    tail = (s["x"][0], s["x"][1], 0.06)
                    head = (s["x"][0] + 0.9*tdir[0], s["x"][1] + 0.9*tdir[1], 0.06)
                    p.addUserDebugLine(tail, head, [1,0.5,0], 3, 0)

            # optional HUD (hidden by default)
            if s["hud_id"] is not None:
                p.removeUserDebugItem(s["hud_id"])
                s["hud_id"] = None
            s["hud_id"] = add_text_if(show_hud,
                                      f"mode={mode}  U={U:.4f}  speed={s['speed']:.2f}x  [X]=quit  [r/R]=restart  [Space]=pause  [ / ]=speed",
                                      [-4.9, 3.6, 0.02], [0,0,0], 1.2, 0.2)

            # LM detection → hold → enable escape (paused 시 카운트다운도 멈춤)
            if (stuck and not s["planner"].allow_escape and s["hold"] < 0):
                s["hold"] = hold_stuck_steps
                # yellow marker (not text)
                vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.07, rgbaColor=[1.0,0.85,0.0,1])
                cs = p.createCollisionShape(p.GEOM_SPHERE, radius=0.07)
                p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cs, baseVisualShapeIndex=vs,
                                  basePosition=(s["x"][0], s["x"][1], 0.06))

            if s["hold"] >= 0:
                add_text_if(show_hud,
                            f"LOCAL MINIMUM — enabling ESCAPE in {s['hold']} steps",
                            [-4.7, 3.2, 0.02], [0.7,0.1,0.1], 1.2, 0.2)
                s["hold"] -= 1
                if s["hold"] == 0:
                    s["planner"].allow_escape = True

            # move state forward
            s["x"] = x_next

            # reach goal → freeze (but keep app running)
            if v_len(v_sub(s["x"], goal_xy)) < s["tol_goal"]:
                s["frozen"] = True
                add_text_if(show_hud,
                            "Goal reached. [Space]=pause/resume, [r/R]=restart, [X]=quit",
                            [-4.9, 2.9, 0.02], [0.1,0.5,0.1], 1.2, 4.0)

        # keep stepping for GUI responsiveness (even when paused/frozen)
        # Speed control: sleep inversely to speed factor
        sleep_dt = 1.0 / (SIM_HZ_BASE * s["speed"])
        p.stepSimulation()
        time.sleep(sleep_dt)

# ------------------------ main ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Use PyBullet GUI (manual quit/restart/pause/speed).")
    parser.add_argument("--gif", action="store_true", help="(Headless) Save animated GIF.")
    parser.add_argument("--hud", action="store_true", help="Show on-screen HUD text (hidden by default).")
    args = parser.parse_args()

    # Local-minimum-prone scenario (close to your figure)
    start = (-4.2, -0.9)
    goal  = ( 4.0,  2.7)
    obstacles = [
        {"c": (0.25, -0.05), "r": 0.80, "color": (0.8, 0.1, 0.1, 0.96)},
        {"c": (1.20,  0.10), "r": 0.80, "color": (0.8, 0.1, 0.1, 0.96)},
        {"c": (0.70,  0.95), "r": 0.80, "color": (0.8, 0.1, 0.1, 0.96)},
    ]

    # Planner params (LM first, then escape)
    zeta, d_star = 0.8, 1.2
    eta,  Q_star = 220.0, 1.6
    step_size    = 0.04
    hold_stuck_steps = 120

    # Connect
    cid = -1
    if args.gui:
        try:
            cid = p.connect(p.GUI)
        except Exception:
            cid = -1
    if cid < 0:
        cid = p.connect(p.DIRECT)  # headless fallback
        args.gui = False

    if args.gui:
        run_gui_loop(start, goal, obstacles,
                     zeta, d_star, eta, Q_star,
                     step_size, hold_stuck_steps,
                     show_hud=args.hud)
        if p.isConnected(): p.disconnect()
    else:
        run_headless_episode(start, goal, obstacles,
                             zeta, d_star, eta, Q_star,
                             step_size, hold_stuck_steps,
                             gif_name="apbp_demo.gif")
        p.disconnect()

if __name__ == "__main__":
    main()
