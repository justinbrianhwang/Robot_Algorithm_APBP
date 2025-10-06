# -*- coding: utf-8 -*-
"""
APBP Parking — 3 points (rigid triangle) → 3 targets
- 장애물 없음, 매력 퍼텐셜만.
- 3점이 '한 차'로 동시에 움직이며 (x,y,theta)만 갱신.
- 도착점 3개를 시작 삼각형의 '합동 복사본'으로 만들 수 있어
  수학적으로 정확히 도킹 가능.
- 목표 근처에서 최소제곱 강체변환(Kabsch-2D)으로 스냅해 미세 오차 제거.

조작:
  - [X] 창 닫기 : 종료
  - r / R        : 리셋
  - Space        : 일시정지/재개
  - [ / ]        : 속도 ↓ / ↑
"""

import math, time
import itertools

# ---------- PyBullet ----------
import pybullet as p
import pybullet_data

# 키 입력 플래그(버전 호환)
KEY_WAS_TRIGGERED = getattr(p, "KEY_WAS_TRIGGERED", 1 << 0)
KEY_SPACE    = getattr(p, "B3G_SPACE", 32)
KEY_LBRACKET = ord('[')
KEY_RBRACKET = ord(']')

# ---------- 작은 벡터 유틸 ----------
def v_add(a,b): return (a[0]+b[0], a[1]+b[1])
def v_sub(a,b): return (a[0]-b[0], a[1]-b[1])
def v_len(a):   return math.hypot(a[0], a[1])

def rot(theta):
    c,s = math.cos(theta), math.sin(theta)
    return ((c,-s),(s,c))

def rot_apply(R,a):
    return (R[0][0]*a[0] + R[0][1]*a[1], R[1][0]*a[0] + R[1][1]*a[1])

# ---------- 합동 목표 생성 & 스냅 ----------
def make_congruent_goals_from_pose(starts, tx, ty, theta_rad):
    """시작 3점을 '중심 기준'으로 회전+이동한 합동 복사본 생성."""
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
    """
    몸체 고정점 L(바디좌표) ↔ 목표점 G(월드) 최소제곱 강체변환 (회전+이동, 스케일X).
    반환: (T_opt, theta_opt)
    """
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

# ---------- APBP: 강체 3점 ----------
class RigidAPBPParking:
    """
    Xi = T + R(theta) * Li   (Li: 바디고정점, i=0..2)
    U = sum 0.5*k*||Xi - Gi||^2
    -> 순힘 F, 순토크 tau 로 (T,theta) gradient 하강
    """
    def __init__(self, start_pts, goal_pts,
                 k_att=1.0, alpha=0.030, beta=0.060,
                 tol_point=0.005, step_limit=8000,
                 dynamic_assignment=False,
                 snap_when_close=True, snap_eps=0.018):
        self.N = 3
        self.G  = [tuple(g) for g in goal_pts]
        self.k_att = k_att

        # 시작 자세 & 바디좌표
        cx = sum(x for x,_ in start_pts)/self.N
        cy = sum(y for _,y in start_pts)/self.N
        p01 = (start_pts[1][0]-start_pts[0][0], start_pts[1][1]-start_pts[0][1])
        self.theta = math.atan2(p01[1], p01[0])
        self.T = (cx, cy)
        R0T = rot(-self.theta)
        self.L = [rot_apply(R0T, (x-cx, y-cy)) for (x,y) in start_pts]

        # 게인(적응형)
        self.alpha0, self.beta0 = alpha, beta
        self.alpha,  self.beta  = alpha, beta

        self.tol_point = tol_point
        self.step_limit = step_limit
        self.dynamic_assignment = dynamic_assignment
        self.snap_when_close = snap_when_close
        self.snap_eps = snap_eps
        self.assign = (0,1,2)  # 고정 매칭(합동이면 그대로 OK)
        self.t = 0

        # 초기 평균오차(적응 게인 스케일 기준)
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
            if sse < best_sse:
                best_sse, best = sse, perm
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
        # 목표 근처에서 게인을 줄여 잔떨림 방지
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

        # (x,y,theta) 업데이트
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

# ---------- 월드 구성 ----------
def create_world(starts, goals):
    p.resetSimulation()
    p.setGravity(0,0,0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane, -1, rgbaColor=[0.95,0.95,0.95,1])

    def add_marker(xy, color, radius=0.075, z=0.06):
        vs = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        cs = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        return p.createMultiBody(0, cs, vs, [xy[0], xy[1], z])

    robots = [add_marker(s, (0.05,0.05,0.05,1.0), 0.075) for s in starts]
    goals  = [add_marker(g, (0.2,1.0,0.2,1.0), 0.085)   for g in goals]

    # 격자/프레임(가시성)
    for a,b in [([-1,-1,0],[13,-1,0]), ([13,-1,0],[13,13,0]),
                ([13,13,0],[-1,13,0]), ([-1,13,0],[-1,-1,0])]:
        p.addUserDebugLine(a,b,[0.7,0.7,0.7],1,0)
    return robots, goals

# ---------- 실행 ----------
def main():
    # 시작 삼각형(차의 3점)
    starts = [(0.0,0.0), (3.0,0.0), (0.0,3.0)]

    # ▶ 합동 목표: (tx,ty)로 이동 + θ(deg) 회전한 복사본
    tx, ty, deg = 7.6, 9.2, 20.0
    goals = make_congruent_goals_from_pose(starts, tx, ty, math.radians(deg))

    # PyBullet GUI 연결(불가 시 DIRECT)
    cid = -1
    try: cid = p.connect(p.GUI)
    except Exception: cid = -1
    if cid < 0:
        cid = p.connect(p.DIRECT)

    robots, _ = create_world(starts, goals)
    p.resetDebugVisualizerCamera(cameraDistance=8.0, cameraYaw=30, cameraPitch=-40,
                                 cameraTargetPosition=[6.0,6.0,0.0])

    planner = RigidAPBPParking(starts, goals,
                               alpha=0.030, beta=0.060,
                               tol_point=0.003, step_limit=8000,
                               dynamic_assignment=False,   # 합동이므로 굳이 필요 X
                               snap_when_close=True, snap_eps=0.015)

    SIM_HZ = 120.0
    speed  = 1.0
    prev0 = None

    while p.isConnected():
        # 키 입력
        keys = p.getKeyboardEvents()
        if keys:
            if ((ord('r') in keys and (keys[ord('r')] & KEY_WAS_TRIGGERED)) or
                (ord('R') in keys and (keys[ord('R')] & KEY_WAS_TRIGGERED))):
                robots, _ = create_world(starts, goals)
                planner = RigidAPBPParking(starts, goals,
                                           alpha=0.030, beta=0.060,
                                           tol_point=0.003, step_limit=8000,
                                           dynamic_assignment=False,
                                           snap_when_close=True, snap_eps=0.015)
                prev0 = None
            if (KEY_LBRACKET in keys) and (keys[KEY_LBRACKET] & KEY_WAS_TRIGGERED):
                speed = max(0.1, speed/1.25)
            if (KEY_RBRACKET in keys) and (keys[KEY_RBRACKET] & KEY_WAS_TRIGGERED):
                speed = min(8.0, speed*1.25)
            if (KEY_SPACE in keys) and (keys[KEY_SPACE] & KEY_WAS_TRIGGERED):
                # 간단 토글: speed 0 ↔ 이전 속도
                speed = 0.0 if speed>0 else 1.0

        if speed > 0:
            X, done = planner.step()
            for i,rb in enumerate(robots):
                p.resetBasePositionAndOrientation(rb, (X[i][0],X[i][1],0.06), [0,0,0,1])
            # 삼각형 윤곽선 + 짧은 궤적
            A,B,C = (X[0][0],X[0][1],0.06), (X[1][0],X[1][1],0.06), (X[2][0],X[2][1],0.06)
            for a,b in [(A,B),(B,C),(C,A)]: p.addUserDebugLine(a,b,[0,0,0],2,0)
            if prev0 is not None: p.addUserDebugLine(prev0, A, [0,0,0],1,0)
            prev0 = A
            if done:
                print("Perfect docking (within tolerance).")
                speed = 0.0  # 정지

        p.stepSimulation()
        time.sleep(1.0/(SIM_HZ*max(speed,0.1)))

if __name__ == "__main__":
    main()
