# -*- coding: utf-8 -*-
"""
APBP Parking — 3 points (rigid triangle) with TAKEOFF
- 원본 인터페이스/인수/조작 그대로.
- UAV는 바닥에서 출발 → 목표 근접 시 이륙해 공중에서 도킹 후 정지.
- 2D (x,y,theta) 경로계획 + z-시각화만 동적 보간.

조작:
  - [X] 창 닫기 : 종료
  - r / R        : 리셋
  - Space        : 일시정지/재개
  - [ / ]        : 속도 ↓ / ↑
"""

import os, math, time, tempfile, itertools
import pybullet as p
import pybullet_data

# ---------- 높이/색/이륙파라미터 ----------
Z_START = 0.06     # 바닥(살짝 띄워 그림자/깜빡임 방지)
Z_GOAL  = 3.0      # 최종 공중 고도
ASCENT_START = 0.35  # 진행도 u(0~1): 이 값 이후부터 상승 시작
ASCENT_END   = 0.90  # 진행도 u(0~1): 이 값에서 목표 고도 도달

RGBA_UAV  = (0.15, 0.35, 1.00, 0.85)  # UAV 파랑(살짝 투명)
RGBA_GOAL = (0.20, 0.90, 0.30, 0.45)  # 목표 초록(투명)

# 키 입력 플래그(버전 호환)
KEY_WAS_TRIGGERED = getattr(p, "KEY_WAS_TRIGGERED", 1 << 0)
KEY_SPACE    = getattr(p, "B3G_SPACE", 32)
KEY_LBRACKET = ord('[')
KEY_RBRACKET = ord(']')

# ---------- 벡터 유틸 ----------
def v_add(a,b): return (a[0]+b[0], a[1]+b[1])
def v_sub(a,b): return (a[0]-b[0], a[1]-b[1])
def v_len(a):   return math.hypot(a[0], a[1])

def rot(theta):
    c,s = math.cos(theta), math.sin(theta)
    return ((c,-s),(s,c))

def rot_apply(R,a):
    return (R[0][0]*a[0] + R[0][1]*a[1], R[1][0]*a[0] + R[1][1]*a[1])

def clamp01(x): return 0.0 if x<0.0 else (1.0 if x>1.0 else x)
def smoothstep(u):  # 0~1에서 3u^2-2u^3 부드러운 보간
    u = clamp01(u); return u*u*(3.0 - 2.0*u)

def remap_clamp(x, a, b):
    if b<=a: return 1.0
    return clamp01((x - a) / (b - a))

# ---------- 합동 목표 생성 (원본과 동일) ----------
def make_congruent_goals_from_pose(starts, tx, ty, theta_rad):
    cx = sum(x for x,_ in starts)/3.0
    cy = sum(y for _,y in starts)/3.0
    R = rot(theta_rad)
    goals = []
    for (x,y) in starts:
        rel = (x-cx, y-cy)
        goals.append((tx + R[0][0]*rel[0] + R[0][1]*rel[1],
                      ty + R[1][0]*rel[0] + R[1][1]*rel[1]))
    return goals

def best_rigid_transform_2d(L, G):
    N = len(L)
    cxL = sum(l[0] for l in L)/N; cyL = sum(l[1] for l in L)/N
    cxG = sum(g[0] for g in G)/N; cyG = sum(g[1] for g in G)/N
    Lc = [(l[0]-cxL, l[1]-cyL) for l in L]
    Gc = [(g[0]-cxG, g[1]-cyG) for g in G]
    Sxx = sum(Lc[i][0]*Gc[i][0] + Lc[i][1]*Gc[i][1] for i in range(N))
    Sxy = sum(Lc[i][0]*Gc[i][1] - Lc[i][1]*Gc[i][0] for i in range(N))
    theta = math.atan2(Sxy, Sxx)
    R = rot(theta)
    RmLc = (R[0][0]*cxL + R[0][1]*cyL, R[1][0]*cxL + R[1][1]*cyL)
    T = (cxG - RmLc[0], cyG - RmLc[1])
    return T, theta

# ---------- APBP (원본과 동일 시그니처/로직) ----------
class RigidAPBPParking:
    def __init__(self, start_pts, goal_pts,
                 k_att=1.0, alpha=0.030, beta=0.060,
                 tol_point=0.005, step_limit=8000,
                 dynamic_assignment=False,
                 snap_when_close=True, snap_eps=0.018):
        self.N = 3
        self.G  = [tuple(g) for g in goal_pts]
        self.k_att = k_att

        cx = sum(x for x,_ in start_pts)/self.N
        cy = sum(y for _,y in start_pts)/self.N
        p01 = (start_pts[1][0]-start_pts[0][0], start_pts[1][1]-start_pts[0][1])
        self.theta = math.atan2(p01[1], p01[0])
        self.T = (cx, cy)
        R0T = rot(-self.theta)
        self.L = [rot_apply(R0T, (x-cx, y-cy)) for (x,y) in start_pts]

        self.alpha0, self.beta0 = alpha, beta
        self.alpha,  self.beta  = alpha, beta
        self.tol_point = tol_point
        self.step_limit = step_limit
        self.dynamic_assignment = dynamic_assignment
        self.snap_when_close = snap_when_close
        self.snap_eps = snap_eps
        self.assign = (0,1,2)
        self.t = 0

        X0 = self.world_points()
        self.err0 = sum(v_len((X0[i][0]-self.G[i][0], X0[i][1]-self.G[i][1])) for i in range(self.N))/self.N

    def world_points(self):
        Rw = rot(self.theta)
        return [v_add(self.T, rot_apply(Rw, Li)) for Li in self.L]

    def _best_assignment(self, X):
        best = None; best_sse = 1e18
        for perm in itertools.permutations(range(self.N)):
            sse = 0.0
            for i,gi in enumerate(perm):
                dx = X[i][0]-self.G[gi][0]; dy = X[i][1]-self.G[gi][1]
                sse += dx*dx+dy*dy
            if sse < best_sse: best_sse, best = sse, perm
        return best

    def _forces(self, X):
        F=(0.0,0.0); tau=0.0
        for i,Xi in enumerate(X):
            Gi = self.G[self.assign[i]]
            fi = (-self.k_att*(Xi[0]-Gi[0]), -self.k_att*(Xi[1]-Gi[1]))
            F = (F[0]+fi[0], F[1]+fi[1])
            ri = (Xi[0]-self.T[0], Xi[1]-self.T[1])
            tau += ri[0]*fi[1] - ri[1]*fi[0]
        return F, tau

    def _adapt_gains(self, avg_err):
        s = max(0.15, min(1.0, avg_err / (0.35*self.err0 + 1e-9)))
        self.alpha = self.alpha0 * s
        self.beta  = self.beta0  * s

    def _snap(self):
        Gperm = [self.G[self.assign[i]] for i in range(self.N)]
        Topt, thopt = best_rigid_transform_2d(self.L, Gperm)
        self.T, self.theta = Topt, thopt

    def step(self):
        X = self.world_points()
        if self.dynamic_assignment:
            self.assign = self._best_assignment(X)

        F, tau = self._forces(X)
        avg_err = sum(v_len((X[i][0]-self.G[self.assign[i]][0],
                             X[i][1]-self.G[self.assign[i]][1])) for i in range(self.N))/self.N
        self._adapt_gains(avg_err)

        self.T = (self.T[0] + self.alpha*F[0], self.T[1] + self.alpha*F[1])
        self.theta += self.beta * tau
        self.t += 1

        Xn = self.world_points()
        if self.snap_when_close and avg_err < self.snap_eps:
            self._snap()
            Xn = self.world_points()

        done = all(v_len((Xn[i][0]-self.G[self.assign[i]][0],
                          Xn[i][1]-self.G[self.assign[i]][1])) < self.tol_point
                   for i in range(self.N))
        return Xn, done

# ---------- 채워진 삼각형 메쉬 ----------
def _write_triangle_obj(local_pts, path):
    with open(path, "w") as f:
        for (x,y) in local_pts:
            f.write(f"v {x:.9f} {y:.9f} 0.0\n")
        f.write("f 1 2 3\n")
        f.write("f 3 2 1\n")  # 양면

def spawn_filled_triangle(local_pts, base_pos_xyz, yaw_rad, rgba):
    tmpdir = tempfile.gettempdir()
    obj_path = os.path.join(tmpdir, "uav_triangle_takeoff.obj")
    _write_triangle_obj(local_pts, obj_path)
    vis = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=obj_path,
                              meshScale=[1,1,1], rgbaColor=list(rgba))
    body = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1,
                             baseVisualShapeIndex=vis,
                             basePosition=[base_pos_xyz[0], base_pos_xyz[1], base_pos_xyz[2]],
                             baseOrientation=p.getQuaternionFromEuler([0,0,yaw_rad]))
    return body

# ---------- 월드 구성(시작=바닥 / 목표=공중) ----------
def create_world(starts, goals):
    p.resetSimulation()
    p.setGravity(0,0,0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane, -1, rgbaColor=[0.95,0.95,0.95,1])

    def add_marker(xy, color, radius=0.060, z=Z_START):
        vs = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        cs = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        return p.createMultiBody(0, cs, vs, [xy[0], xy[1], z])

    # 시작점(바닥), 목표점(공중)
    robots = [add_marker(s, (0.05,0.05,0.05,0.9), 0.050, Z_START) for s in starts]
    goals_m = [add_marker(g, (0.2,1.0,0.2,0.6), 0.055, Z_GOAL) for g in goals]

    # 프레임
    for a,b in [([-1,-1,0],[13,-1,0]), ([13,-1,0],[13,13,0]),
                ([13,13,0],[-1,13,0]), ([-1,13,0],[-1,-1,0])]:
        p.addUserDebugLine(a,b,[0.7,0.7,0.7],1,0)

    return robots, goals_m

# ---------- 실행 ----------
def main():
    # 시작 삼각형(이등변 예: 직각이등변)
    starts = [(0.0,0.0), (3.0,0.0), (0.0,3.0)]

    # ▶ 합동 목표: (tx,ty) 이동 + θ(deg) 회전 (원본과 동일 인수)
    tx, ty, deg = 7.6, 9.2, 20.0
    goals = make_congruent_goals_from_pose(starts, tx, ty, math.radians(deg))

    try: cid = p.connect(p.GUI)
    except Exception: cid = -1
    if cid < 0: cid = p.connect(p.DIRECT)

    robots, _ = create_world(starts, goals)
    p.resetDebugVisualizerCamera(cameraDistance=8.0, cameraYaw=30, cameraPitch=-40,
                                 cameraTargetPosition=[6.0,6.0,0.0])

    planner = RigidAPBPParking(starts, goals,
                               alpha=0.030, beta=0.060,
                               tol_point=0.003, step_limit=8000,
                               dynamic_assignment=False,
                               snap_when_close=True, snap_eps=0.015)

    # 목표 삼각형(공중)의 정확한 위치/자세
    T_goal, yaw_goal = best_rigid_transform_2d(planner.L, goals)

    # UAV는 "바닥"에서 시작
    uav_tri  = spawn_filled_triangle(planner.L, (planner.T[0], planner.T[1], Z_START),
                                     yaw_rad=planner.theta, rgba=RGBA_UAV)
    goal_tri = spawn_filled_triangle(planner.L, (T_goal[0], T_goal[1], Z_GOAL),
                                     yaw_rad=yaw_goal, rgba=RGBA_GOAL)

    SIM_HZ = 120.0
    speed  = 1.0
    prevA = None

    while p.isConnected():
        # --- 키 입력 ---
        keys = p.getKeyboardEvents()
        if keys:
            if ((ord('r') in keys and (keys[ord('r')] & KEY_WAS_TRIGGERED)) or
                (ord('R') in keys and (keys[ord('R')] & KEY_WAS_TRIGGERED))):
                if uav_tri is not None:  p.removeBody(uav_tri)
                if goal_tri is not None: p.removeBody(goal_tri)
                robots, _ = create_world(starts, goals)
                planner = RigidAPBPParking(starts, goals,
                                           alpha=0.030, beta=0.060,
                                           tol_point=0.003, step_limit=8000,
                                           dynamic_assignment=False,
                                           snap_when_close=True, snap_eps=0.015)
                T_goal, yaw_goal = best_rigid_transform_2d(planner.L, goals)
                uav_tri  = spawn_filled_triangle(planner.L, (planner.T[0], planner.T[1], Z_START),
                                                 yaw_rad=planner.theta, rgba=RGBA_UAV)
                goal_tri = spawn_filled_triangle(planner.L, (T_goal[0], T_goal[1], Z_GOAL),
                                                 yaw_rad=yaw_goal, rgba=RGBA_GOAL)
                prevA = None
            if (KEY_LBRACKET in keys) and (keys[KEY_LBRACKET] & KEY_WAS_TRIGGERED):
                speed = max(0.1, speed/1.25)  # 느리게
            if (KEY_RBRACKET in keys) and (keys[KEY_RBRACKET] & KEY_WAS_TRIGGERED):
                speed = min(8.0, speed*1.25)  # 빠르게
            if (KEY_SPACE in keys) and (keys[KEY_SPACE] & KEY_WAS_TRIGGERED):
                speed = 0.0 if speed>0 else 1.0

        if speed > 0:
            X, done = planner.step()

            # ---- 진행도 기반 고도 보간 ----
            # u: 0(시작)~1(도착) 진행도
            avg_err = sum(v_len((X[i][0]-goals[i][0], X[i][1]-goals[i][1])) for i in range(3))/3.0
            u = 1.0 - (avg_err / (planner.err0 + 1e-12))
            # s: 이륙 구간에서만 0→1로 부드럽게
            s = smoothstep(remap_clamp(u, ASCENT_START, ASCENT_END))
            z_now = (1.0 - s)*Z_START + s*Z_GOAL

            # 점 마커(3점) 위치 갱신: z_now 사용
            for i,rb in enumerate(robots):
                p.resetBasePositionAndOrientation(rb, (X[i][0], X[i][1], z_now), [0,0,0,1])

            # UAV 채워진 삼각형 위치/자세 갱신
            p.resetBasePositionAndOrientation(
                uav_tri,
                (planner.T[0], planner.T[1], z_now),
                p.getQuaternionFromEuler([0,0,planner.theta])
            )

            # 윤곽선 & 궤적(현재 z에 맞춰)
            A,B,C = (X[0][0],X[0][1],z_now), (X[1][0],X[1][1],z_now), (X[2][0],X[2][1],z_now)
            for a,b in [(A,B),(B,C),(C,A)]:
                p.addUserDebugLine(a,b,[0.02,0.10,0.60],2,0)
            if prevA is not None:
                p.addUserDebugLine(prevA, A, [0,0,0],1,0)
            prevA = A

            if done:
                print("Perfect docking (within tolerance).")
                speed = 0.0  # 도착 시 정지

        p.stepSimulation()
        time.sleep(1.0/(SIM_HZ*max(speed,0.1)))

if __name__ == "__main__":
    main()
