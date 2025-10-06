import pybullet as p
import pybullet_data
import numpy as np
import random, time, math

# -------------------------
# 시뮬레이션 파라미터
# -------------------------
GUI = True
DT = 1.0/240.0
SIM_TIME = 60.0
GRAVITY = -9.81

# 실시간 속도 제어
SPEED_FACTOR     = 1.0
SPEED_STEP       = 1.25
SPEED_MIN, SPEED_MAX = 0.25, 8.0
paused = False

# 원형 경기장(반지름 R) + 벽
ARENA_R        = 6.0
WALL_THICKNESS = 0.18
WALL_HEIGHT    = 1.2
WALL_SEGMENTS  = 64
WALL_OVERLAP   = 1.02

# 로봇(더 크게)
ROBOT_HALF  = [0.40, 0.40, 0.12]  # ≈ 0.8m x 0.8m
ROBOT_MASS  = 12.0
ROBOT_VMAX  = 4.0
ROBOT_RADIUS = max(ROBOT_HALF[0], ROBOT_HALF[1]) * 1.2

# 장애물(구)
SPAWN_RATE       = 2.0     # +/=/−/_ 로 조절
SPAWN_RATE_STEP  = 0.5
SPAWN_RATE_MIN   = 0.0
SPAWN_RATE_MAX   = 12.0
SPAWN_Z          = 6.0
OBS_RADIUS       = 0.22
OBS_MASS         = 1.0

# “가운데도 떨어지게” - 중심 쏠림
SPAWN_CENTER_BIAS = 1.4    # 1.0=면적 균등, >1 중심 쏠림↑
SPAWN_INNER_R     = 0.0

# APBP: Attractive/Repulsive
ZETA = 2.5
ETA  = 1.8
RHO0 = 2.2           # 근접 기반에서의 영향 반경(수평 중심거리)

# --- 근접 기반 가중치 파라미터(자연스러운 회피 핵심) ---
Z_GATE_TAU   = 1.2   # 높이 게이트 길이(↓이면 공이 높이 있을 때 영향 급감)
TCA_TAU      = 0.8   # 최근접시각(time-to-closest-approach) 게이트
AWAY_WEIGHT  = 0.35  # 멀어질 때 최소 반영 비율

KF   = 3.0
DAMP = 0.4

# 로컬 미니마 탈출 서브플래너
STUCK_VELOCITY = 0.15
STUCK_REP_MIN  = 2.0
ROT_ANGLE      = math.pi/2
ROT_STEPS      = 40

rng = random.Random(0)

# -------------------------
# 유틸리티
# -------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def saturate_vec(vx, vy, vmax):
    n = math.hypot(vx, vy)
    if n <= 1e-9: return (0.0, 0.0)
    if n > vmax:
        s = vmax / n
        return (vx*s, vy*s)
    return (vx, vy)

def rot2d(vec, angle):
    c, s = math.cos(angle), math.sin(angle)
    return (c*vec[0] - s*vec[1], s*vec[0] + c*vec[1])

def sample_annulus(inner_r, outer_r, center_bias=1.0):
    """원환(내반지름~외반지름)에서 샘플. >1 중심 쏠림↑, <1 바깥쪽 쏠림↑"""
    u = rng.random() ** center_bias
    r_sq = inner_r*inner_r + (outer_r*outer_r - inner_r*inner_r) * u
    r = math.sqrt(r_sq)
    a = rng.uniform(0, 2*math.pi)
    return r*math.cos(a), r*math.sin(a)

# -------------------------
# PyBullet 초기화
# -------------------------
cid = p.connect(p.GUI if GUI else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setTimeStep(DT)
p.setGravity(0, 0, GRAVITY)
p.setPhysicsEngineParameter(numSolverIterations=60, fixedTimeStep=DT)

plane = p.loadURDF("plane.urdf")

# 로봇
col = p.createCollisionShape(p.GEOM_BOX, halfExtents=ROBOT_HALF)
vis = p.createVisualShape(p.GEOM_BOX, halfExtents=ROBOT_HALF, rgbaColor=[0.1, 0.6, 0.9, 1])
start = [0, 0, ROBOT_HALF[2]]
robot = p.createMultiBody(baseMass=ROBOT_MASS, baseCollisionShapeIndex=col,
                          baseVisualShapeIndex=vis, basePosition=start)
p.changeDynamics(robot, -1, lateralFriction=1.0, rollingFriction=0.001,
                 linearDamping=0.05, angularDamping=0.05)

# 장애물 shape
obs_col = p.createCollisionShape(p.GEOM_SPHERE, radius=OBS_RADIUS)
obs_vis = p.createVisualShape(p.GEOM_SPHERE, radius=OBS_RADIUS, rgbaColor=[0.9, 0.2, 0.2, 1])

# -------------------------
# 원형 벽(정다각형 근사)
# -------------------------
def build_circular_wall(radius, segments, thickness, height, overlap=1.02):
    s = 2.0 * radius * math.tan(math.pi / segments) * overlap
    hx, hy, hz = thickness * 0.5, s * 0.5, height * 0.5
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=[0.5, 0.5, 0.5, 1])
    wall_ids = []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        orn = p.getQuaternionFromEuler([0, 0, theta])  # x=radial, y=tangent
        wid = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col,
                                baseVisualShapeIndex=vis,
                                basePosition=[x, y, hz], baseOrientation=orn)
        p.changeDynamics(wid, -1, lateralFriction=1.0, restitution=0.0)
        wall_ids.append(wid)
    return wall_ids, s

walls, seg_len = build_circular_wall(ARENA_R, WALL_SEGMENTS, WALL_THICKNESS, WALL_HEIGHT, WALL_OVERLAP)

# 스폰 반경(벽/로봇 여유 반영)
SPAWN_MARGIN = WALL_THICKNESS*0.5 + ROBOT_RADIUS + 0.3
SPAWN_R_MAX  = max(0.5, ARENA_R - SPAWN_MARGIN)
SPAWN_R_MIN  = max(0.0, min(SPAWN_INNER_R, SPAWN_R_MAX-0.1))

# -------------------------
# 스폰/관리
# -------------------------
obstacles = []  # {id, r}
def spawn_obstacle():
    x, y = sample_annulus(SPAWN_R_MIN, SPAWN_R_MAX, SPAWN_CENTER_BIAS)
    z = SPAWN_Z
    oid = p.createMultiBody(baseMass=OBS_MASS, baseCollisionShapeIndex=obs_col,
                            baseVisualShapeIndex=obs_vis, basePosition=[x, y, z])
    p.changeDynamics(oid, -1, lateralFriction=0.8, restitution=0.0,
                     linearDamping=0.0, angularDamping=0.0)
    vx = rng.uniform(-0.5, 0.5)
    vy = rng.uniform(-0.5, 0.5)
    p.resetBaseVelocity(oid, [vx, vy, 0], [0,0,0])
    obstacles.append({"id": oid, "r": OBS_RADIUS})

# -------------------------
# 재시작(Reset)
# -------------------------
def reset_sim():
    global t, hits, accum_spawn, rot_timer, rot_dir, paused
    for o in list(obstacles):
        try: p.removeBody(o["id"])
        except: pass
    obstacles.clear()
    p.resetBasePositionAndOrientation(robot, [0, 0, ROBOT_HALF[2]], [0,0,0,1])
    p.resetBaseVelocity(robot, [0,0,0], [0,0,0])
    t = 0.0; hits = 0; accum_spawn = 0.0
    rot_timer = 0; rot_dir = 0; paused = False

# -------------------------
# 키보드 입력
# -------------------------
KEY_LEFT_BRACKET  = ord('[')
KEY_RIGHT_BRACKET = ord(']')
KEY_SPACE         = p.B3G_SPACE if hasattr(p, "B3G_SPACE") else ord(' ')
KEY_PLUS, KEY_EQUAL = ord('+'), ord('=')
KEY_MINUS, KEY_UNDERSCORE = ord('-'), ord('_')
KEY_r, KEY_R = ord('r'), ord('R')

def handle_keyboard():
    global SPEED_FACTOR, paused, SPAWN_RATE
    events = p.getKeyboardEvents()
    if not events: return
    if KEY_SPACE in events and (events[KEY_SPACE] & p.KEY_WAS_TRIGGERED):
        paused = not paused
    if KEY_LEFT_BRACKET in events and (events[KEY_LEFT_BRACKET] & p.KEY_WAS_TRIGGERED):
        SPEED_FACTOR = clamp(SPEED_FACTOR / SPEED_STEP, SPEED_MIN, SPEED_MAX)
    if KEY_RIGHT_BRACKET in events and (events[KEY_RIGHT_BRACKET] & p.KEY_WAS_TRIGGERED):
        SPEED_FACTOR = clamp(SPEED_FACTOR * SPEED_STEP, SPEED_MIN, SPEED_MAX)
    if (KEY_PLUS in events and (events[KEY_PLUS] & p.KEY_WAS_TRIGGERED)) or \
       (KEY_EQUAL in events and (events[KEY_EQUAL] & p.KEY_WAS_TRIGGERED)):
        SPAWN_RATE = clamp(SPAWN_RATE + SPAWN_RATE_STEP, SPAWN_RATE_MIN, SPAWN_RATE_MAX)
    if (KEY_MINUS in events and (events[KEY_MINUS] & p.KEY_WAS_TRIGGERED)) or \
       (KEY_UNDERSCORE in events and (events[KEY_UNDERSCORE] & p.KEY_WAS_TRIGGERED)):
        SPAWN_RATE = clamp(SPAWN_RATE - SPAWN_RATE_STEP, SPAWN_RATE_MIN, SPAWN_RATE_MAX)
    if (KEY_r in events and (events[KEY_r] & p.KEY_WAS_TRIGGERED)) or \
       (KEY_R in events and (events[KEY_R] & p.KEY_WAS_TRIGGERED)):
        reset_sim()

# -------------------------
# 제어 루프
# -------------------------
goal = np.array([0.0, 0.0])  # 중앙 유지
t = 0.0
accum_spawn = 0.0
hits = 0

rot_timer = 0
rot_dir = 0  # -1 or +1

# HUD 하단 배치
hud_x, hud_y, hud_z = 0.0, -ARENA_R + 0.05, 0.02
hud_id = p.addUserDebugText("", [hud_x, hud_y, hud_z], [0,0,0], textSize=1.2)
def update_hud():
    status = "⏸ PAUSED" if paused else f"{SPEED_FACTOR:.2f}x"
    txt = (f"Time {t:5.2f}s | Obstacles {len(obstacles)} | Hits {hits} | Speed {status} | Spawn {SPAWN_RATE:.2f}/s\n"
           f"Controls: '[' slow  ']' fast · Space pause/resume · (+/=) more  (−/_) fewer · R restart")
    p.addUserDebugText(txt, [hud_x, hud_y, hud_z], [0,0,0], textSize=1.2, replaceItemUniqueId=hud_id)

while t < SIM_TIME:
    handle_keyboard()

    if not paused:
        # 스폰
        accum_spawn += DT * SPAWN_RATE
        while accum_spawn >= 1.0:
            spawn_obstacle()
            accum_spawn -= 1.0

        # 로봇 상태
        (rx, ry, rz), _ = p.getBasePositionAndOrientation(robot)
        (rvx, rvy, rvz) = p.getBaseVelocity(robot)[0]
        r = np.array([rx, ry])

        # APBP 힘 계산 (근접 기반 Repulsive)
        F_att = ZETA * (goal - r)
        F_rep = np.array([0.0, 0.0])
        nearest = None
        nearest_rho = 1e9

        for o in list(obstacles):
            oid = o["id"]
            (ox, oy, oz), _ = p.getBasePositionAndOrientation(oid)
            (ovx, ovy, ovz) = p.getBaseVelocity(oid)[0]

            o_xy = np.array([ox, oy])
            diff = r - o_xy
            rho = float(np.linalg.norm(diff)) + 1e-9
            if rho < nearest_rho:
                nearest_rho = rho
                nearest = (o_xy, diff)

            # 상대 속도(수평)
            vrel = np.array([rvx - ovx, rvy - ovy])
            v2 = float(np.dot(vrel, vrel))
            if v2 > 1e-12:
                t_star = max(0.0, - float(np.dot(diff, vrel)) / v2)  # 최근접 시각
            else:
                t_star = 1e9  # 거의 정지 상대

            # 가중치: 높이, TCA, 접근/이탈
            w_z = math.exp(- max(0.0, oz - o["r"]) / Z_GATE_TAU)
            w_t = math.exp(- t_star / TCA_TAU)
            approaching = (np.dot(diff, vrel) < 0.0)
            w_a = 1.0 if approaching else AWAY_WEIGHT
            w = w_z * w_t * w_a

            # Repulsive (수평 중심거리 기준)
            if rho < RHO0:
                coeff = ETA * w * (1.0/rho - 1.0/RHO0) / (rho**3)
                F_rep += coeff * diff

            # 바닥에 닿으면 제거
            if oz <= o["r"] + 1e-3:
                p.removeBody(oid)
                obstacles.remove(o)

        # 로컬 미니마 회전 서브플래너
        F_total = F_att + F_rep
        rep_mag = float(np.linalg.norm(F_rep))
        vel_mag = math.hypot(rvx, rvy)

        if rot_timer > 0:
            F_att_rot = rot2d((F_att[0], F_att[1]), rot_dir * ROT_ANGLE)
            F_total = np.array([F_att_rot[0], F_att_rot[1]]) + F_rep
            rot_timer -= 1
        else:
            if vel_mag < STUCK_VELOCITY and rep_mag > STUCK_REP_MIN and nearest is not None:
                o_xy, diff = nearest
                att_vec = (goal - r)
                cross = diff[0]*att_vec[1] - diff[1]*att_vec[0]
                rot_dir = 1 if cross >= 0 else -1
                rot_timer = ROT_STEPS
                F_att_rot = rot2d((F_att[0], F_att[1]), rot_dir * ROT_ANGLE)
                F_total = np.array([F_att_rot[0], F_att_rot[1]]) + F_rep

        # 속도 명령 → 포화
        vd = KF * F_total - DAMP * np.array([rvx, rvy])
        vx, vy = saturate_vec(vd[0], vd[1], ROBOT_VMAX)
        p.resetBaseVelocity(robot, [vx, vy, 0], [0,0,0])

        # 충돌 체크
        for o in list(obstacles):
            cps = p.getContactPoints(bodyA=robot, bodyB=o["id"])
            if cps:
                hits += 1
                p.removeBody(o["id"])
                obstacles.remove(o)

        p.stepSimulation()
        t += DT

    update_hud()
    if GUI:
        time.sleep((DT / SPEED_FACTOR) if not paused else 0.02)

print("Simulation done. Hits =", hits)
p.disconnect()
