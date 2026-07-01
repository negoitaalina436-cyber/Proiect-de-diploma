import time
import math
import threading
import io
import collections
from collections import deque

import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

import board
import busio
from adafruit_servokit import ServoKit
import RPi.GPIO as GPIO

from flask import Flask, Response, jsonify, render_template_string
try:
    from mpu6050 import mpu6050 as MPU6050Class
except ImportError:
    MPU6050Class = None

# Modul de intoarcere la statie (MQTT + RSSI WiFi)
from robot_homing import home_to_station
import rtc_module

# Global configurable parameters

CONFIDENCE_THRESHOLD = 0.15

BALL_COLOR_HSV_LOW  = (35, 40, 80)    # calibrat empiric cu tool calibrare_hsv
BALL_COLOR_HSV_HIGH = (75, 255, 255)
BALL_COLOR_MIN_AREA = 30
IMG_SIZE = 640           # marit la 640 pentru detectie mai precisa
                         # (camera 640x480, mingi mici/departate vizibile)
IMG_SIZE_PATROL = 640

# CONSTANTE OPTICE SI GEOMETRIE
# Camera: Picamera2 NoIR (OV5647) la 640x480

FOCAL_LENGTH_PX = 320.0   # calibrat empiric pentru detectie minge
X_SCALE         = 1.26    # corectie lateral (distorsiune lentila NoIR)
BALL_DIAMETER_M = 0.067   # diametru fizic minge de tenis

COLLECTOR_DIAMETER_M = 0.20
COLLECTOR_GROUP_M    = 0.15
COLLECTOR_LENGTH_M   = 0.15

STOP_DISTANCE_M      = 0.50   # opreste cand minge mai aproape de Z = 50cm
COLLECT_EXTRA_TIME_S = 2.0    # timp extra forward dupa atingerea STOP_DISTANCE
                              # (compenseaza erorile la distanta scurta)

# PINI HARDWARE
PIN_RPWM = 33    # BTS7960 forward
PIN_LPWM = 32    # BTS7960 backward
PWM_FREQ_HZ = 1000

STEERING_CHANNEL = 0    # ServoKit channel directie roti
CAMERA_CHANNEL   = 7    # ServoKit channel servo camera
CAMERA_CENTER    = 90.0

LED_RED    = 11
LED_YELLOW = 13
LED_GREEN  = 15
CAPACITY_MAX  = 10
CAPACITY_WARN = 5

MPU_ADDR = 0x69   # MPU6050 giroscop pe I2C (adresa non-default)

# VIRAJ, giroscop primar, timp ca backup
# Factori de viraj separati pentru stanga / dreapta, compenseaza asimetria
# mecanica a directiei Ackermann (jocul in servo, geometria roti).
TURN_FACTOR_LEFT  = 0.70   # calibrat empiric
TURN_FACTOR_RIGHT = 0.70   # calibrat empiric

USE_TIME_BASED_TURN      = False   # giroscop primar; True = doar pe timp
DEGREES_PER_SECOND_RIGHT = 5.50    # calibrat empiric (backup time-based)
DEGREES_PER_SECOND_LEFT  = 7.50    # calibrat empiric
TURN_SPEED_DUTY = 65.0
GYRO_NORM_DEG   = 30.0

# UNGHIURI SERVO DIRECTIE
# Center = 60, regla mecanica pentru roti drepte
# Safe Left = 30, Safe Right = 90, limite empirice (blocaj mecanic peste)
# Ofset simetric: -30 stanga, +30 dreapta
STEERING_LEFT_ANGLE   = 10.0    # limita absoluta servo (nu se ajunge in practica)
STEERING_CENTER_ANGLE = 60.0    # rotile drepte
STEERING_RIGHT_ANGLE  = 88.0    # limita absoluta servo

STEERING_SAFETY_MARGIN = 3.0
STEERING_SAFE_LEFT  = 43.0   # calibrat empiric - limita mecanica reala
STEERING_SAFE_RIGHT = 85.0   # calibrat empiric

# Camera servo: 170 stanga, 10 dreapta (camera intoarsa fizic)
CAMERA_LEFT  = 170.0
CAMERA_RIGHT = 10.0

# SONAR HC-SR04 (frontal)
SONAR_TRIG_PIN  = 18
SONAR_ECHO_PIN  = 16
SONAR_STOP_M    = 0.50    # opreste cand obstacol < 50cm
SONAR_BLIND_M   = 0.00
SONAR_CONSEC    = 3       # citiri consecutive pentru confirm
SONAR_TIMEOUT_S = 0.025

PREVIEW_WINDOW  = "Robot Camera"
PREVIEW_ENABLED = False

PID_KP = 1.20
PID_KI = 0.00
PID_KD = 0.00

MAX_STEER_DEG = 35.0
PID_ERROR_ALPHA = 0.60

HIGH_DUTY        = 100.0
EXPLORE_DUTY     = 70.0
PATROL_DUTY      = 90.0
COLLECT_DUTY     = 30.0

PRE_STEER_THRESH = 0.25

COURT_WIDTH_M  = 8.23
COURT_LENGTH_M = 23.77
GRID_SPACING_M = 1.2

_state_lock = threading.Lock()
_shared = {
    "frame_jpg":       None,
    "balls_seen":      0,
    "balls_collected": 0,
    "steering_angle":  STEERING_CENTER_ANGLE,
    "speed_duty":      0.0,
    "direction":       "stop",
    "distance_m":      0.0,
    "phase":           "init",
    "sonar_m":         99.0,
    "brightness":      0.0,
    "robot_x":         0.0,
    "robot_y":         0.0,
    "patrol_wp":       0,
    "patrol_total":    0,
}

def state_set(**kwargs):
    with _state_lock:
        _shared.update(kwargs)

def state_get(key):
    with _state_lock:
        return _shared[key]

_servo_kit   = None
_pwm_r       = None
_pwm_l       = None
_current_dir = "stop"
_current_duty = 0.0

def clamp(x, x_min, x_max):
    return max(x_min, min(x_max, x))


def init_actuators():
    global _servo_kit, _pwm_r, _pwm_l, _current_dir, _current_duty
    mode = GPIO.getmode()
    if mode is None:
        GPIO.setmode(GPIO.BOARD)
    elif mode != GPIO.BOARD:
        GPIO.cleanup()
        GPIO.setmode(GPIO.BOARD)
    GPIO.setup(PIN_RPWM, GPIO.OUT)
    GPIO.setup(PIN_LPWM, GPIO.OUT)
    _pwm_r = GPIO.PWM(PIN_RPWM, PWM_FREQ_HZ)
    _pwm_l = GPIO.PWM(PIN_LPWM, PWM_FREQ_HZ)
    _pwm_r.start(0)
    _pwm_l.start(0)
    _current_dir  = "stop"
    _current_duty = 0.0
    i2c = busio.I2C(board.SCL, board.SDA)
    _servo_kit = ServoKit(channels=16, i2c=i2c)
    try:
        _servo_kit.servo[STEERING_CHANNEL].angle = STEERING_CENTER_ANGLE
        _servo_kit.servo[CAMERA_CHANNEL].angle   = CAMERA_CENTER
    except Exception as e:
        print(f"Warning: could not center steering servo: {e}")
    GPIO.setup(SONAR_TRIG_PIN, GPIO.OUT)
    GPIO.setup(SONAR_ECHO_PIN, GPIO.IN)
    GPIO.setup([LED_RED, LED_YELLOW, LED_GREEN], GPIO.OUT)
    GPIO.output(LED_RED,    GPIO.LOW)
    GPIO.output(LED_YELLOW, GPIO.LOW)
    GPIO.output(LED_GREEN,  GPIO.HIGH)
    GPIO.output(SONAR_TRIG_PIN, False)
    time.sleep(0.03)


def set_motor_pwm(direction: str, duty: float):
    global _current_dir, _current_duty
    if _pwm_r is None or _pwm_l is None:
        return
    duty = clamp(float(duty), 0.0, 100.0)
    if direction == "forward":
        _pwm_l.ChangeDutyCycle(0.0)
        _pwm_r.ChangeDutyCycle(duty)
    elif direction == "backward":
        _pwm_r.ChangeDutyCycle(0.0)
        _pwm_l.ChangeDutyCycle(duty)
    else:
        _pwm_r.ChangeDutyCycle(0.0)
        _pwm_l.ChangeDutyCycle(0.0)
        direction, duty = "stop", 0.0
    _current_dir, _current_duty = direction, duty
    state_set(direction=direction, speed_duty=duty)


def move_robot(direction: str, duration: float = 0.0, duty: float = 60.0):
    set_motor_pwm(direction, duty)
    if duration and duration > 0:
        time.sleep(duration)
        set_motor_pwm("stop", 0.0)


_last_servo_angle = 55.0

def set_camera_angle(angle: float):
    angle = clamp(angle, min(CAMERA_LEFT, CAMERA_RIGHT), max(CAMERA_LEFT, CAMERA_RIGHT))
    if _servo_kit is not None:
        try:
            _servo_kit.servo[CAMERA_CHANNEL].angle = angle
        except Exception as e:
            print(f"[Camera servo] Eroare: {e}")
    return angle


def camera_center():
    set_camera_angle(CAMERA_CENTER)


# Limite plauzibilitate pentru detectiile YOLO de la distanta.
# La depth > 2m, mingea are foarte putini pixeli si calculul X (lateral)
# se amplifica enorm cu micile erori de bbox. La |lateral|>1.5m, mingea
# ar fi la >45deg pe parti unde FOV-ul nu o vede oricum.
# La depth < 0.12m mingea ar umple tot cadrul - fizic imposibil (camera e
# la 30cm inaltime), deci e fals pozitiv (reflexie, colector in cadru, etc).
MIN_BALL_DEPTH_M    = 0.12
MAX_BALL_DEPTH_M    = 1.5
MAX_BALL_LATERAL_M  = 1.5


def _is_plausible_ball(X, Z):
    """Filtreaza detectii implauzibile (probabil zgomot/erori de measurement)."""
    if Z < MIN_BALL_DEPTH_M:
        print(f"    [Filtru] Ignor detectie: depth={Z:.2f}m < {MIN_BALL_DEPTH_M}m (prea aproape, fals pozitiv)")
        return False
    if Z > MAX_BALL_DEPTH_M:
        print(f"    [Filtru] Ignor detectie: depth={Z:.2f}m > {MAX_BALL_DEPTH_M}m (prea departe)")
        return False
    if abs(X) > MAX_BALL_LATERAL_M:
        print(f"    [Filtru] Ignor detectie: lateral={X:+.2f}m > {MAX_BALL_LATERAL_M}m (prea lateral)")
        return False
    return True


# Distanta-prag pentru early exit la scan: daca robotul gaseste o minge
# DREPT INAINTE (camera la centru) mai aproape de aceasta distanta, opreste
# scanarea, e cazul ideal pentru navigare directa fara viraj.
SCAN_EARLY_EXIT_DEPTH_M = 1.0


def camera_scan_arc(picam2, model, sports_ball_cls_id, fx, fy, cx0, cy0, patrol=None):
    """Scanare YOLO in 3 pozitii camera: CENTRU, DREAPTA, STANGA.

    Strategie:
      - CENTRU: daca detecteaza minge - returneaza direct coordonatele (Z, X)
      - DREAPTA/STANGA: daca detecteaza minge - seteaza flag 'needs_reverse'
        cu directia (robotul va da cu spatele in directia opusa pentru
        a aduce mingea in fata, apoi rescaneaza)

    Filtre: cy < 50 (zona fisheye superioara, false positives) eliminate.
    """
    scan_positions = [CAMERA_CENTER, CAMERA_RIGHT, CAMERA_LEFT]
    print("[Camera] Scanare rapida YOLO...")

    for cam_angle in scan_positions:
        set_camera_angle(cam_angle)
        pos_str = ("CENTRU" if cam_angle == CAMERA_CENTER
                   else "DREAPTA" if cam_angle == CAMERA_RIGHT else "STANGA")
        print(f"  [Scan {pos_str}] Servomotor camera la {cam_angle:.0f}deg")
        time.sleep(0.3)  # pauza ca sa se stabilizeze camera dupa miscare

        frame_bgr = capture_corrected(picam2)

        results = model(
            frame_bgr,
            imgsz=IMG_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            iou=0.35,
            max_det=3,
            verbose=False,
        )

        balls_at_this_angle = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                if int(box.cls[0]) != sports_ball_cls_id:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cxb = (x1 + x2) // 2
                cyb = (y1 + y2) // 2
                wpx = x2 - x1

                p3d = compute_3d_position(cxb, cyb, wpx, fx, fy, cx0, cy0)
                if p3d is None:
                    continue
                X, Y, Z = p3d

                # Folosim coordonatele directe doar pentru camera CENTRATA
                # Pentru lateral, doar marcam ca exista minge in directia aceea
                if cam_angle == CAMERA_CENTER:
                    if not _is_plausible_ball(X, Z):
                        continue
                    print(f"  [Centru] Minge: cx={cxb}px Z={Z:.2f}m lat={X:+.2f}m")
                    balls_at_this_angle.append({
                        "cx": cxb, "cy": cyb, "Z": Z, "X_world": X,
                        "box": (x1, y1, x2, y2),
                        "cam_angle": cam_angle,
                        "needs_reverse": False,
                    })
                else:
                    # Lateral, doar flag, nu coordonate
                    # Filtru: detectii la marginea de sus (cy<50) sunt false
                    # pozitive din distorsiunea fisheye/iluminat
                    if cyb < 50:
                        print(f"  [Lateral] Ignor cy={cyb}px (margine sus, fals pozitiv)")
                        continue
                    direction = "DREAPTA" if cam_angle == CAMERA_RIGHT else "STANGA"
                    print(f"  [{direction}] Minge detectata lateral (cx={cxb}px cy={cyb}px), "
                          f"flag needs_reverse")
                    balls_at_this_angle.append({
                        "cx": cxb, "cy": cyb, "Z": 0.0, "X_world": 0.0,
                        "box": (x1, y1, x2, y2),
                        "cam_angle": cam_angle,
                        "needs_reverse": True,
                        "reverse_direction": direction,
                    })

        if balls_at_this_angle:
            camera_center()
            if cam_angle == CAMERA_CENTER:
                print(f"[Camera] {len(balls_at_this_angle)} minge(i) la centru")
            else:
                print(f"[Camera] Minge detectata lateral, "
                      f"trebuie sa dau cu spatele pana vede verde")
            return balls_at_this_angle

    camera_center()
    print("[Camera] Scan complet, nicio minge")
    return []

def set_steering_angle(angle: float):
    global _last_servo_angle
    angle = clamp(angle, STEERING_SAFE_LEFT, STEERING_SAFE_RIGHT)
    _last_servo_angle = angle
    if _servo_kit is not None:
        try:
            _servo_kit.servo[STEERING_CHANNEL].angle = angle
            time.sleep(0.02)
            _servo_kit.servo[STEERING_CHANNEL].angle = angle
        except Exception as e:
            print(f"Servo error: {e}")
    state_set(steering_angle=angle)
    return angle


def stop_and_center():
    set_motor_pwm("stop", 0.0)
    set_steering_angle(STEERING_CENTER_ANGLE)


def update_status_led(n_collected: int):
    try:
        GPIO.output(LED_RED,    GPIO.LOW)
        GPIO.output(LED_YELLOW, GPIO.LOW)
        GPIO.output(LED_GREEN,  GPIO.LOW)
        if n_collected >= CAPACITY_MAX:
            GPIO.output(LED_RED, GPIO.HIGH)
        elif n_collected >= CAPACITY_WARN:
            GPIO.output(LED_YELLOW, GPIO.HIGH)
        else:
            GPIO.output(LED_GREEN, GPIO.HIGH)
    except Exception:
        pass


PASSIVE_SUSPECT_S = 6.0   # robotul trebuie sa fie suspect 4s inainte de testul activ
                          # (marit de la 2.5s, la mers drept lung, gz=0 dadea false positive)
PASSIVE_YAW_DELTA_THRESH = 0.25
ACTIVE_TEST_STEER_DEG = 18.0
ACTIVE_TEST_DURATION_S = 0.45
ACTIVE_TEST_YAW_THRESH = 0.30
ACTIVE_TEST_COOLDOWN_S = 2.0
ACTIVE_TEST_DUTY = 55.0
GZ_IO_DEAD_THRESH = 0.005
STUCK_SAMPLE_COUNT = 25
STUCK_SAMPLE_DT = 0.04
# Rate limit pentru check_stuck: nu rulam fereastra gyro (1s blocant)
# mai des de aceasta perioada.
CHECK_STUCK_PERIOD = 2.0


def _read_gz_for_stuck():
    try:
        return _mpu.get_gyro_data()["z"] - _gz_off
    except Exception:
        return None


def _stuck_gyro_window():
    readings = []
    for _ in range(STUCK_SAMPLE_COUNT):
        v = _read_gz_for_stuck()
        if v is not None:
            readings.append(v)
        time.sleep(STUCK_SAMPLE_DT)
    if len(readings) < max(5, STUCK_SAMPLE_COUNT // 2):
        return None, None, None, None
    import statistics as _stats
    gz_std = _stats.stdev(readings) if len(readings) > 1 else 0.0
    gz_mean_abs = sum(abs(v) for v in readings) / len(readings)
    gz_signed = sum(readings) / len(readings)
    yaw_delta = sum(readings) * STUCK_SAMPLE_DT
    return gz_std, gz_mean_abs, gz_signed, yaw_delta


def _stuck_measure_yaw_for(duration_s):
    yaw = 0.0
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        v = _read_gz_for_stuck()
        if v is not None:
            yaw += v * STUCK_SAMPLE_DT
        time.sleep(STUCK_SAMPLE_DT)
    return yaw


def _active_stuck_test():
    print("[Stuck] Test activ: viraj stanga/dreapta...")
    set_motor_pwm("stop", 0.0)
    time.sleep(0.15)
    set_steering_angle(STEERING_CENTER_ANGLE - ACTIVE_TEST_STEER_DEG)
    set_motor_pwm("forward", ACTIVE_TEST_DUTY)
    yaw_left = _stuck_measure_yaw_for(ACTIVE_TEST_DURATION_S)
    set_motor_pwm("stop", 0.0)
    time.sleep(0.15)
    set_steering_angle(STEERING_CENTER_ANGLE + ACTIVE_TEST_STEER_DEG)
    set_motor_pwm("forward", ACTIVE_TEST_DUTY)
    yaw_right = _stuck_measure_yaw_for(ACTIVE_TEST_DURATION_S)
    set_motor_pwm("stop", 0.0)
    time.sleep(0.15)
    set_steering_angle(STEERING_CENTER_ANGLE)
    print(f"[Stuck] yaw_left={yaw_left:+.2f}deg yaw_right={yaw_right:+.2f}deg")
    if abs(yaw_left) < ACTIVE_TEST_YAW_THRESH and abs(yaw_right) < ACTIVE_TEST_YAW_THRESH:
        return True
    return False


def check_stuck(state, suspect_threshold=None):
    """Verifica daca robotul e blocat. Rate-limited: nu ruleaza fereastra
    gyro mai des de o data la CHECK_STUCK_PERIOD secunde, pentru ca fereastra
    blocheaza firul ~1s (25 sample-uri x 40ms).
    
    suspect_threshold: prag custom in secunde pentru cat trebuie sa fie suspect
    inainte de testul activ. Daca None, foloseste PASSIVE_SUSPECT_S (default 6s).
    """
    threshold = suspect_threshold if suspect_threshold is not None else PASSIVE_SUSPECT_S
    if _mpu is None:
        return False
    now = time.monotonic()
    # Rate limit: nu rulam fereastra gyro mai des de la 2s la 2s
    last_check = state.get('last_window_check', 0.0)
    if now - last_check < CHECK_STUCK_PERIOD:
        return False
    state['last_window_check'] = now

    res = _stuck_gyro_window()
    if res[0] is None:
        state['suspect_since'] = None
        return False
    gz_std, gz_mean, gz_signed, yaw_delta = res
    now = time.monotonic()
    if gz_std < GZ_IO_DEAD_THRESH and gz_mean < GZ_IO_DEAD_THRESH:
        if state['suspect_since'] is None:
            state['suspect_since'] = now
    elif abs(yaw_delta) <= PASSIVE_YAW_DELTA_THRESH:
        if state['suspect_since'] is None:
            state['suspect_since'] = now
    else:
        state['suspect_since'] = None
        return False
    elapsed = now - state['suspect_since']
    can_test = now - state['last_active_test'] >= ACTIVE_TEST_COOLDOWN_S
    if elapsed >= threshold and can_test:
        state['last_active_test'] = now
        is_stuck = _active_stuck_test()
        state['suspect_since'] = None
        return is_stuck
    return False


def cleanup_actuators():
    global _pwm_r, _pwm_l, _servo_kit
    stop_sonar()
    time.sleep(0.15)
    stop_and_center()
    try:
        if _pwm_r: _pwm_r.stop()
        if _pwm_l: _pwm_l.stop()
    except Exception:
        pass
    _pwm_r = _pwm_l = None
    try:
        GPIO.cleanup()
    except Exception:
        pass
    _servo_kit = None


def read_sonar_once() -> float:
    GPIO.output(SONAR_TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(SONAR_TRIG_PIN, False)
    deadline = time.monotonic() + SONAR_TIMEOUT_S
    while GPIO.input(SONAR_ECHO_PIN) == 0:
        if time.monotonic() > deadline:
            return 99.0
    pulse_start = time.monotonic()
    deadline = time.monotonic() + SONAR_TIMEOUT_S
    while GPIO.input(SONAR_ECHO_PIN) == 1:
        if time.monotonic() > deadline:
            return 99.0
    pulse_end = time.monotonic()
    distance = (pulse_end - pulse_start) * 34300 / 2.0 / 100.0
    return round(distance, 4)


_sonar_stop = threading.Event()

def sonar_loop():
    global _sonar_recent
    while not _sonar_stop.is_set():
        try:
            d = read_sonar_once()
            if d < SONAR_BLIND_M:
                d = 99.0
            state_set(sonar_m=d)
            _sonar_recent.append(d)
            if len(_sonar_recent) > 5:
                _sonar_recent.pop(0)
        except Exception as e:
            if not _sonar_stop.is_set():
                print(f"[Sonar] error: {e}")
        time.sleep(0.10)


def start_sonar():
    _sonar_stop.clear()
    t = threading.Thread(target=sonar_loop, daemon=True)
    t.start()
    print("[Sonar] Firul de fundal a pornit.")


def stop_sonar():
    _sonar_stop.set()


_sonar_recent = []

def sonar_blocked() -> bool:
    if len(_sonar_recent) < 3:
        return False
    valid = [d for d in _sonar_recent if d < 90.0]
    if len(valid) < 3:
        return False
    obstacles = sum(1 for d in valid if SONAR_BLIND_M <= d < SONAR_STOP_M)
    return obstacles >= SONAR_CONSEC


def preview_loop(picam2_ref):
    if not PREVIEW_ENABLED:
        return
    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PREVIEW_WINDOW, 640, 480)
    _exp_state = {}
    while True:
        try:
            frame_bgr = capture_corrected(picam2_ref)
            mean_bright, exp_us, gain = adapt_exposure(picam2_ref, frame_bgr, _exp_state)
            state_set(brightness=round(mean_bright, 1))
            sonar_val = state_get("sonar_m")
            sonar_txt = f"Sonar: {sonar_val:.2f}m" if sonar_val < 90 else "Sonar: --"
            exp_txt   = f"Exp: {int(exp_us)}us  Gain: {gain:.1f}x  Bright: {mean_bright:.0f}"
            cv2.putText(frame_bgr, sonar_txt, (8, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_bgr, exp_txt, (8, 475),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1, cv2.LINE_AA)
            cv2.imshow(PREVIEW_WINDOW, frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except Exception:
            break
    cv2.destroyWindow(PREVIEW_WINDOW)


class PIDSteering:
    def __init__(self, kp=PID_KP, ki=PID_KI, kd=PID_KD, alpha=PID_ERROR_ALPHA):
        self.kp = kp; self.ki = ki; self.kd = kd; self.alpha = alpha
        self.reset()

    def reset(self, seed_x: float = None, frame_width: float = 640.0):
        self._integral  = 0.0
        self._prev_time = None
        if seed_x is not None:
            raw = clamp((seed_x - frame_width/2.0) / (frame_width/2.0), -1.0, 1.0)
        else:
            raw = 0.0
        self._filtered_err  = raw
        self._prev_filt_err = raw

    def compute(self, target_x: float, frame_width: float) -> float:
        now      = time.monotonic()
        center_x = frame_width / 2.0
        raw_err  = clamp((target_x - center_x) / center_x, -1.0, 1.0)
        filt_err = (1.0 - self.alpha) * raw_err + self.alpha * self._filtered_err
        self._filtered_err = filt_err
        if self._prev_time is None:
            self._prev_filt_err = filt_err
            self._prev_time     = now
        dt = clamp(now - self._prev_time, 0.005, 0.5)
        self._prev_time = now
        if self.ki != 0.0:
            self._integral += filt_err * dt
            self._integral  = clamp(self._integral, -1.0, 1.0)
        derivative = (filt_err - self._prev_filt_err) / dt
        self._prev_filt_err = filt_err
        pid_out = self.kp * filt_err + self.ki * self._integral + self.kd * derivative
        pid_out = clamp(pid_out, -1.0, 1.0)
        if pid_out >= 0:
            angle = STEERING_CENTER_ANGLE + pid_out * (STEERING_RIGHT_ANGLE - STEERING_CENTER_ANGLE)
        else:
            angle = STEERING_CENTER_ANGLE + pid_out * (STEERING_CENTER_ANGLE - STEERING_LEFT_ANGLE)
        angle = clamp(angle, STEERING_LEFT_ANGLE, STEERING_RIGHT_ANGLE)
        print(f"[PID] raw={raw_err:+.3f} filt={filt_err:+.3f} deriv={derivative:+.3f} - {angle:.1f}deg")
        return set_steering_angle(angle)


pid = PIDSteering()


class PathRecorder:
    def __init__(self):
        self._moves = []
    def record(self, angle, duration, duty):
        self._moves.append((angle, duration, duty))
    def return_to_start(self):
        if not self._moves:
            return
        print("[Return] Retracing path to start...")
        state_set(phase="returning")
        for angle, duration, duty in reversed(self._moves):
            mirrored = 2 * STEERING_CENTER_ANGLE - angle
            set_steering_angle(mirrored)
            move_robot("backward", duration=duration, duty=duty)
            time.sleep(0.03)
        stop_and_center()
        print("[Return] Arrived at start.")
        self._moves.clear()
    def clear(self):
        self._moves.clear()


recorder = PathRecorder()


EXP_MIN_US   =  5_000
EXP_MAX_US   = 50_000
GAIN_MIN     =  1.0
GAIN_MAX     = 12.0
TARGET_BRIGHT_LOW  = 100
TARGET_BRIGHT_HIGH = 180
EXP_STEP_FRAC = 0.20


def init_camera():
    """Initializeaza camera Picamera2 NoIR la 640x480, 30fps.
    Setari critice: AWB off + ColourGains (1.0, 2.5), tenta calda care
    permite modelului YOLO custom sa detecteze mingile (conf >= 0.65)."""
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        controls={"FrameRate": 30.0},
    )
    picam2.configure(config)
    picam2.start()
    picam2.set_controls({
        "AeEnable":   True,            # auto-expunere initiala
        "AwbEnable":  True,            # AWB auto activ
        # "ColourGains": (1.0, 2.5),   # dezactivat, AWB se ocupa
        "Brightness": -0.3,
        "Contrast":   1.3,
        "Saturation": 1.1,
    })
    time.sleep(1.5)
    meta = picam2.capture_metadata()
    init_exp  = clamp(meta.get("ExposureTime", 20_000), EXP_MIN_US, EXP_MAX_US)
    init_gain = clamp(meta.get("AnalogueGain",  4.0),  GAIN_MIN,   GAIN_MAX)
    picam2.set_controls({
        "AeEnable":     False,
        "ExposureTime": int(init_exp),
        "AnalogueGain": float(init_gain),
    })
    print(f"[Camera] Predare AE: expunere={init_exp}us  gain={init_gain:.1f}  AWB=ON")
    time.sleep(0.3)
    return picam2


def adapt_exposure(picam2, frame_bgr, state: dict):
    if not state:
        meta = picam2.capture_metadata()
        state["exp_us"] = float(meta.get("ExposureTime", 20_000))
        state["gain"]   = float(meta.get("AnalogueGain",  4.0))
    grey  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean  = float(grey.mean())
    exp  = state["exp_us"]
    gain = state["gain"]
    if mean < TARGET_BRIGHT_LOW:
        deficit = (TARGET_BRIGHT_LOW - mean) / TARGET_BRIGHT_LOW
        factor  = 1.0 + EXP_STEP_FRAC * deficit
        exp = exp * factor
        if exp > EXP_MAX_US:
            exp  = EXP_MAX_US
            gain = clamp(gain * factor, GAIN_MIN, GAIN_MAX)
    elif mean > TARGET_BRIGHT_HIGH:
        excess  = (mean - TARGET_BRIGHT_HIGH) / TARGET_BRIGHT_HIGH
        factor  = 1.0 - EXP_STEP_FRAC * excess
        gain = gain * factor
        if gain < GAIN_MIN:
            gain = GAIN_MIN
            exp  = clamp(exp * factor, EXP_MIN_US, EXP_MAX_US)
    exp  = clamp(exp,  EXP_MIN_US, EXP_MAX_US)
    gain = clamp(gain, GAIN_MIN,   GAIN_MAX)
    state["exp_us"] = exp
    state["gain"]   = gain
    picam2.set_controls({"ExposureTime": int(exp), "AnalogueGain": float(gain)})
    return mean, exp, gain


def capture_corrected(picam2):
    """Capture frame BGR direct din camera, fara post-procesare.
    Modelul YOLO custom (best.pt) detecteaza mingi cu acest pipeline brut
    + ColourGains (1.0, 2.5) setat in init_camera (conf >= 0.65 in conditii bune).
    """
    frame_rgb = picam2.capture_array("main")
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return frame_bgr


def compute_3d_position(cx, cy, px_width, fx, fy, cx0, cy0):
    """Estimare 3D din bbox YOLO, formulele pinhole standard.
      Z = D_real * fx / w_px         (adancime)
      X = (cx - cx0) * Z / fx * X_SCALE  (lateral, corectat distorsiune)
      Y = (cy - cy0) * Z / fy            (vertical, nefolosit)
    Returneaza (X, Y, Z) in metri, sau None daca bbox invalid.
    """
    if px_width <= 0:
        return None
    Z = (BALL_DIAMETER_M * fx) / float(px_width)
    X = ((cx - cx0) * Z / fx) * X_SCALE
    Y = (cy - cy0) * Z / fy
    return X, Y, Z


# Viteza reala empirica a robotului la duty 100% (calibrata cu cronometru)
REAL_SPEED_M_S = 0.100

def calculate_travel_time(distance_m: float) -> float:
    """Timp necesar parcurgerii unei distante, la viteza de patrulare."""
    speed = REAL_SPEED_M_S * (PATROL_DUTY / 100.0)
    t = distance_m / speed
    print(f"[Travel] {distance_m:.3f}m / {speed:.3f}m/s = {t:.2f}s max")
    return t


def encode_frame_jpg(frame_bgr) -> bytes:
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


def annotate_frame(frame_rgb, balls):
    out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for b in balls:
        x1, y1, x2, y2 = b["box"]
        color = (0, 220, 80)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ball Z={b['Z']:.2f}m"
        cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    with _state_lock:
        phase    = _shared["phase"]
        angle    = _shared["steering_angle"]
        duty     = _shared["speed_duty"]
        collected = _shared["balls_collected"]
    sonar  = _shared.get("sonar_m", 99.0)
    bright = _shared.get("brightness", 0.0)
    sonar_str = f"{sonar:.2f}m" if sonar < 90 else "--"
    hud = (f"{phase.upper()}  steer={angle:.0f}deg  duty={duty:.0f}%  "
           f"collected={collected}  sonar={sonar_str}  bright={bright:.0f}")
    cv2.putText(out, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return out


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ro"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tennis Ball Robot</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
    color: #2d3748;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    min-height: 100vh;
    padding: 2rem 1rem;
  }
  .container { max-width: 1200px; margin: 0 auto; }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2rem;
    padding: 1rem 1.5rem;
    background: white;
    border-radius: 16px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
  }
  .logo {
    display: flex; align-items: center; gap: 0.75rem;
  }
  .logo-icon {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 4px 8px rgba(16, 185, 129, 0.3);
  }
  .logo h1 {
    font-size: 1.4rem; color: #1a202c;
    letter-spacing: -0.02em;
  }
  .logo .subtitle {
    font-size: 0.8rem; color: #718096;
    margin-top: 2px;
  }
  .status-badge {
    padding: 0.5rem 1rem;
    background: #ecfdf5;
    color: #047857;
    border-radius: 20px;
    font-size: 0.85rem; font-weight: 600;
    display: flex; align-items: center; gap: 0.5rem;
  }
  .status-dot {
    width: 8px; height: 8px;
    background: #10b981;
    border-radius: 50%;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  .main-grid {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 1.5rem;
  }
  @media (max-width: 768px) {
    .main-grid { grid-template-columns: 1fr; }
  }
  .stream-card {
    background: white;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
  }
  .stream-header {
    padding: 1rem 1.5rem;
    border-bottom: 1px solid #e2e8f0;
    display: flex; align-items: center; justify-content: space-between;
  }
  .stream-header h2 {
    font-size: 1rem; color: #4a5568; font-weight: 600;
  }
  #stream {
    width: 100%; display: block;
    background: #1a202c;
  }
  .stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }
  .stat-card {
    background: white;
    border-radius: 16px;
    padding: 1.25rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    transition: transform 0.2s;
  }
  .stat-card:hover { transform: translateY(-2px); }
  .stat-card.wide { grid-column: 1 / -1; }
  .stat-label {
    font-size: 0.75rem;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
    margin-bottom: 0.5rem;
    display: flex; align-items: center; gap: 0.4rem;
  }
  .stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1a202c;
    letter-spacing: -0.02em;
    line-height: 1;
  }
  .stat-unit { font-size: 1rem; color: #718096; font-weight: 400; }
  .stat-sub {
    font-size: 0.8rem;
    color: #718096;
    margin-top: 0.4rem;
  }
  .progress-bar {
    height: 6px; background: #e2e8f0;
    border-radius: 3px; overflow: hidden;
    margin-top: 0.6rem;
  }
  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
    border-radius: 3px;
    transition: width 0.3s;
  }
  .phase-pill {
    display: inline-block;
    padding: 0.3rem 0.7rem;
    border-radius: 10px;
    font-size: 0.85rem;
    font-weight: 600;
    background: #dbeafe;
    color: #1e40af;
  }
  .phase-pill.patrolling { background: #dbeafe; color: #1e40af; }
  .phase-pill.collecting { background: #fef3c7; color: #92400e; }
  .phase-pill.homing { background: #f3e8ff; color: #6b21a8; }
  .phase-pill.idle { background: #f1f5f9; color: #475569; }
  .dir-arrow { font-size: 0.9rem; margin-right: 0.3rem; }
  .icon { font-size: 1rem; }
</style></head>
<body>
<div class="container">
  <header>
    <div class="logo">
      <div class="logo-icon">🤖</div>
      <div>
        <h1>Tennis Ball Robot</h1>
        <div class="subtitle">Sistem autonom de colectare</div>
      </div>
    </div>
    <div class="status-badge">
      <span class="status-dot"></span>
      <span>LIVE</span>
    </div>
  </header>

  <div class="main-grid">
    <div class="stream-card">
      <div class="stream-header">
        <h2>📹 Camera live</h2>
        <span class="stat-sub" id="stream-info">640×480</span>
      </div>
      <img id="stream" src="/stream" alt="Camera feed">
    </div>

    <div class="stats-grid">
      <div class="stat-card wide">
        <div class="stat-label"><span class="icon">⚙️</span> Faza curentă</div>
        <div class="phase-pill" id="phase">--</div>
        <div class="stat-sub" id="sonar-badge"></div>
      </div>

      <div class="stat-card">
        <div class="stat-label"><span class="icon">🎾</span> Mingi colectate</div>
        <div class="stat-value" id="collected">0<span class="stat-unit">/10</span></div>
        <div class="progress-bar"><div class="progress-fill" id="collected-bar" style="width:0%"></div></div>
      </div>

      <div class="stat-card">
        <div class="stat-label"><span class="icon">🎯</span> Direcție</div>
        <div class="stat-value"><span id="steer-val">60</span><span class="stat-unit">°</span></div>
        <div class="stat-sub" id="steer-label">centru</div>
      </div>

      <div class="stat-card wide">
        <div class="stat-label"><span class="icon">🏍️</span> Motor</div>
        <div class="stat-value"><span id="duty-val">0</span><span class="stat-unit">%</span></div>
        <div class="stat-sub"><span class="dir-arrow" id="dir-arrow">⏸</span><span id="dir-label">oprit</span></div>
        <div class="progress-bar"><div class="progress-fill" id="duty-bar" style="width:0%"></div></div>
      </div>

      <div class="stat-card wide">
        <div class="stat-label"><span class="icon">💡</span> Luminozitate cadru</div>
        <div class="stat-value" id="bright-val">--</div>
      </div>
    </div>
  </div>
</div>

<script>
async function poll() {
  try {
    const r = await fetch('/status'); const d = await r.json();
    const phaseEl = document.getElementById('phase');
    phaseEl.textContent = d.phase;
    phaseEl.className = 'phase-pill ' + (d.phase || 'idle');

    const balls = d.balls_collected || 0;
    document.getElementById('collected').innerHTML = balls + '<span class="stat-unit">/10</span>';
    document.getElementById('collected-bar').style.width = Math.min(100, balls*10) + '%';

    const steer = d.steering_angle.toFixed(0);
    document.getElementById('steer-val').textContent = steer;
    const center = 60;
    let steerLabel = 'centru';
    if (steer < center - 3) steerLabel = '← stânga (' + Math.abs(steer-center).toFixed(0) + '°)';
    else if (steer > center + 3) steerLabel = 'dreapta - (' + (steer-center).toFixed(0) + '°)';
    document.getElementById('steer-label').textContent = steerLabel;

    const duty = d.speed_duty.toFixed(0);
    document.getElementById('duty-val').textContent = duty;
    document.getElementById('duty-bar').style.width = Math.min(100, duty) + '%';
    const dir = d.direction || 'stopped';
    const arrows = { forward: '▲', backward: '▼', stop: '⏸', stopped: '⏸' };
    document.getElementById('dir-arrow').textContent = arrows[dir] || '⏸';
    const labelMap = { forward: 'înainte', backward: 'înapoi', stop: 'oprit', stopped: 'oprit' };
    document.getElementById('dir-label').textContent = labelMap[dir] || dir;

    const sonarM = d.sonar_m < 90 ? d.sonar_m.toFixed(2)+' m' : '— m';
    document.getElementById('sonar-badge').textContent = 'sonar: ' + sonarM;
    if (d.brightness !== undefined) {
      document.getElementById('bright-val').textContent = Math.round(d.brightness);
    }
  } catch(e) {}
  setTimeout(poll, 400);
}
poll();
</script>
</body></html>
"""

app = Flask(__name__)
import logging
logging.getLogger("werkzeug").setLevel(logging.ERROR)

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route("/status")
def status():
    with _state_lock:
        return jsonify(dict(_shared, frame_jpg=None))

def _gen_frames():
    while True:
        jpg = state_get("frame_jpg")
        if jpg:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(0.03)


def _continuous_stream_thread(picam2_ref):
    """Capture cadre continuu in fundal pentru stream-ul Flask.
    Actualizeaza state_get('frame_jpg') chiar daca robotul nu ruleaza YOLO."""
    while True:
        try:
            frame_rgb = picam2_ref.capture_array("main")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            jpg = encode_frame_jpg(frame_bgr)
            state_set(frame_jpg=jpg)
        except Exception:
            pass
        time.sleep(0.1)   # ~10 fps stream

@app.route("/stream")
def stream():
    return Response(_gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

def start_dashboard(host="0.0.0.0", port=5000):
    t = threading.Thread(target=lambda: app.run(host=host, port=port, threaded=True), daemon=True)
    t.start()
    print(f"[Tablou de bord] Ruleza la http://{host}:{port}")


def group_balls_into_targets(balls: list) -> list:
    remaining = list(balls)
    targets   = []
    used      = set()
    for i, a in enumerate(remaining):
        if i in used:
            continue
        # Mingile cu needs_reverse nu se pot grupa (nu au coordonate reale)
        if a.get("needs_reverse", False):
            targets.append({"cx": float(a["cx"]), "Z": 0.0,
                            "X_world": 0.0,
                            "kind": "single", "balls": [a],
                            "needs_reverse": True,
                            "reverse_direction": a.get("reverse_direction")})
            used.add(i)
            continue

        paired = False
        for j, b in enumerate(remaining):
            if j <= i or j in used:
                continue
            # Nu grupa cu mingi care au needs_reverse
            if b.get("needs_reverse", False):
                continue
            lateral_dist = abs(a.get("X_world", 0.0) - b.get("X_world", 0.0))
            if lateral_dist <= COLLECTOR_GROUP_M:
                mid_cx = (a["cx"] + b["cx"]) / 2.0
                mid_Z  = min(a["Z"], b["Z"])
                mid_X  = (a.get("X_world", 0.0) + b.get("X_world", 0.0)) / 2.0
                targets.append({"cx": mid_cx, "Z": mid_Z, "X_world": mid_X,
                                "kind": "pair", "balls": [a, b],
                                "needs_reverse": False,
                                "reverse_direction": None})
                used.add(i); used.add(j); paired = True
                break
        if not paired and i not in used:
            targets.append({"cx": float(a["cx"]), "Z": a["Z"],
                            "X_world": a.get("X_world", 0.0),
                            "kind": "single", "balls": [a],
                            "needs_reverse": False,
                            "reverse_direction": None})
            used.add(i)
    return targets


def nearest_neighbour_path(targets: list, start_cx: float = 320.0) -> list:
    """Ordoneaza tintele in ordinea distantei reale (X_world, Z) de la robot.
    Robotul incepe la (0, 0). La fiecare pas alege tinta cea mai apropiata
    de pozitia curenta (in metri, nu in pixeli)."""
    if not targets:
        return []
    remaining = list(targets)
    path = []
    cur_x = 0.0   # pozitia curenta a robotului in lume (m)
    cur_z = 0.0
    while remaining:
        best_idx = None; best_dist = None
        for i, t in enumerate(remaining):
            dx = t.get("X_world", 0.0) - cur_x
            dz = t["Z"] - cur_z
            d  = math.sqrt(dx*dx + dz*dz)
            if best_dist is None or d < best_dist:
                best_dist = d; best_idx = i
        chosen = remaining.pop(best_idx)
        path.append(chosen)
        # Robotul ajunge la pozitia tintei
        cur_x = chosen.get("X_world", 0.0)
        cur_z = chosen["Z"]
    return path


def log_scan_results(balls, targets, path):
    print("\n" + "=" * 60)
    print(f"  SCANARE COMPLETA, {len(balls)} minge(i) gasita(e)")
    print("=" * 60)
    for idx, b in enumerate(balls):
        x_w = b.get("X_world", 0.0)
        print(f"  Ball {idx + 1:>2}: screen_x={b['cx']:>3}px  screen_y={b['cy']:>3}px  "
              f"lateral={x_w:+.3f}m  depth={b['Z']:.3f}m")
    print(f"\n  Grupate in {len(targets)} tinta(e):")
    for idx, t in enumerate(targets):
        ball_ids = [balls.index(b) + 1 for b in t["balls"]]
        ids_str  = "+".join(str(i) for i in ball_ids)
        print(f"  Target {idx + 1}: [{t['kind']:>6}]  balls={ids_str:<5}  "
              f"screen_x={t['cx']:>6.1f}px  depth={t['Z']:.3f}m  lateral={t['X_world']:+.3f}m")
    print(f"\n  Ordinea traseului nearest-neighbour:")
    for step, t in enumerate(path):
        ball_ids = [balls.index(b) + 1 for b in t["balls"]]
        ids_str  = "+".join(str(i) for i in ball_ids)
        arrow    = "-" if step < len(path) - 1 else "*"
        print(f"    {arrow} Step {step + 1}: {t['kind']} (ball {ids_str})  "
              f"screen_x={t['cx']:.1f}px  depth={t['Z']:.3f}m")
    print("=" * 60 + "\n")


def build_patrol_grid() -> list:
    cols = max(2, round(COURT_WIDTH_M  / GRID_SPACING_M) + 1)
    rows = max(2, round(COURT_LENGTH_M / GRID_SPACING_M) + 1)
    xs   = [round(i * COURT_WIDTH_M  / (cols - 1), 3) for i in range(cols)]
    ys   = [round(j * COURT_LENGTH_M / (rows - 1), 3) for j in range(rows)]
    wps  = []
    for i, x in enumerate(xs):
        col_ys = ys if i % 2 == 0 else list(reversed(ys))
        for y in col_ys:
            if x == 0.0 and y == 0.0:
                continue
            wps.append((x, y))
    print(f"[Patrula] Grila {cols}x{rows} = {len(wps)} puncte de control "
          f"({COURT_WIDTH_M}m x {COURT_LENGTH_M}m  spatiere={GRID_SPACING_M}m)")
    return wps


class PatrolState:
    def __init__(self):
        self.waypoints = []
        self.visited   = set()
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_hdg = 0.0
    def reset(self, waypoints):
        self.waypoints = waypoints
        self.visited   = set()
        self.robot_x = 0.0; self.robot_y = 0.0; self.robot_hdg = 0.0
        state_set(patrol_wp=0, patrol_total=len(waypoints), robot_x=0.0, robot_y=0.0)
    def update_pos(self, angle_deg, duration, duty):
        speed = REAL_SPEED_M_S * (duty / 100.0)
        dist  = speed * duration
        norm  = (angle_deg - STEERING_CENTER_ANGLE) / (STEERING_RIGHT_ANGLE - STEERING_CENTER_ANGLE)
        self.robot_hdg += norm * math.radians(30)
        self.robot_x   += dist * math.sin(self.robot_hdg)
        self.robot_y   += dist * math.cos(self.robot_hdg)
        state_set(robot_x=round(self.robot_x, 3), robot_y=round(self.robot_y, 3))
    def next_in_sequence(self):
        for i, wp in enumerate(self.waypoints):
            if i not in self.visited:
                return (i, wp)
        return None
    def mark_visited(self, idx):
        self.visited.add(idx)
        state_set(patrol_wp=len(self.visited))
    def all_visited(self):
        return len(self.visited) >= len(self.waypoints)
    def progress(self):
        pct = 100 * len(self.visited) / max(1, len(self.waypoints))
        return (f"{len(self.visited)}/{len(self.waypoints)} ({pct:.0f}%)  "
                f"pos=({self.robot_x:.2f},{self.robot_y:.2f})m")


def has_ball_color(frame_bgr) -> bool:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(BALL_COLOR_HSV_LOW, dtype=np.uint8),
                       np.array(BALL_COLOR_HSV_HIGH, dtype=np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) >= BALL_COLOR_MIN_AREA:
            return True
    return False


def _reverse_50cm():
    """Marsarier pana 0.5m parcursi. NU folosim sonarul (frontal, irelevant
    la mers inapoi). Timeout siguranta 15s."""
    set_motor_pwm("backward", 70.0)
    t0_back = time.monotonic()
    while True:
        dist_back_m = (time.monotonic() - t0_back) * REAL_SPEED_M_S * 0.70
        if dist_back_m >= 0.5: break
        if time.monotonic() - t0_back > 15.0: break
        time.sleep(0.03)
    set_motor_pwm("stop", 0.0)


def drive_to_waypoint(patrol, wx, wy, picam2, model, sports_ball_cls_id,
                      fx, fy, cx0, cy0) -> list:
    """Mers spre un waypoint din traseul de patrulare.
    Cicleaza intre mers drept (DRIVE_DURATION=5s) si scan YOLO (~1s).
    Returneaza lista de mingi confirmate de YOLO (dictionar 'screen_x', 'X', 'Z').
    Daca eroare directie > 70deg, face manevra spate-fata (intoarcere larga).
    """
    dx  = wx - patrol.robot_x
    dy  = wy - patrol.robot_y
    dist = math.hypot(dx, dy)
    if dist < 0.50:
        print(f"[Patrula] Waypoint prea aproape ({dist:.2f}m), skip")
        return []
    desired = math.atan2(dx, dy)
    err     = (desired - patrol.robot_hdg + math.pi) % (2*math.pi) - math.pi
    norm    = clamp(err / math.radians(60), -1.0, 1.0)
    if norm >= 0:
        angle = STEERING_CENTER_ANGLE + norm*(STEERING_RIGHT_ANGLE - STEERING_CENTER_ANGLE)
    else:
        angle = STEERING_CENTER_ANGLE + norm*(STEERING_CENTER_ANGLE - STEERING_LEFT_ANGLE)
    err_deg = math.degrees(err)
    print(f"[Patrula] eroare_directie={err_deg:.1f}deg - servo={angle:.1f}deg")

    if abs(err_deg) > 70.0:
        print(f"[Patrula] Eroare prea mare ({err_deg:.0f}deg), manevra spate-fata")
        if err_deg > 0:
            set_steering_angle(STEERING_LEFT_ANGLE + 5)
        else:
            set_steering_angle(STEERING_RIGHT_ANGLE - 5)
        time.sleep(0.3)
        set_motor_pwm("backward", 80.0)
        time.sleep(1.5)
        set_motor_pwm("stop", 0.0)
        time.sleep(0.3)
        dx = wx - patrol.robot_x
        dy = wy - patrol.robot_y
        desired = math.atan2(dx, dy)
        err = (desired - patrol.robot_hdg + math.pi) % (2*math.pi) - math.pi
        norm = clamp(err / math.radians(60), -1.0, 1.0)
        if norm >= 0:
            angle = STEERING_CENTER_ANGLE + norm*(STEERING_RIGHT_ANGLE - STEERING_CENTER_ANGLE)
        else:
            angle = STEERING_CENTER_ANGLE + norm*(STEERING_CENTER_ANGLE - STEERING_LEFT_ANGLE)
        patrol.robot_hdg = 0.0
        print(f"[Patrula] Dupa manevra: eroare={math.degrees(err):.1f}deg - servo={angle:.1f}deg")

    set_steering_angle(angle)
    travel_time = calculate_travel_time(dist)
    state_set(phase="patrolling")

    turn_time = min(abs(err_deg) / 20.0, 3.0)
    print(f"[Patrula] Viraj in mers: {turn_time:.1f}s  eroare={err_deg:.1f}deg")
    set_motor_pwm("forward", PATROL_DUTY)
    t0_turn = time.monotonic()
    while time.monotonic() - t0_turn < turn_time:
        if sonar_blocked():
            set_motor_pwm("stop", 0.0)
            print(f"[Patrula] Obstacol, dau cu spatele virat 50cm")
            if not hasattr(drive_to_waypoint, "_stuck_dir"):
                drive_to_waypoint._stuck_dir = 1
            drive_to_waypoint._stuck_dir *= -1
            if drive_to_waypoint._stuck_dir > 0:
                set_steering_angle(STEERING_LEFT_ANGLE + 5)
            else:
                set_steering_angle(STEERING_RIGHT_ANGLE - 5)
            _reverse_50cm()
            print("[Patrula] Spatele gata, repornesc virat opus")
            if drive_to_waypoint._stuck_dir > 0:
                set_steering_angle(STEERING_RIGHT_ANGLE - 5)
            else:
                set_steering_angle(STEERING_LEFT_ANGLE + 5)
            set_motor_pwm("forward", PATROL_DUTY)
            time.sleep(2.0)
            set_steering_angle(STEERING_CENTER_ANGLE)
            set_motor_pwm("stop", 0.0)
            patrol.robot_y = 0.0
            return []
        time.sleep(0.03)
        # NU actualizam pozitia in timpul virajului, e imprecis, robotul
        # se misca pe arc nu pe linie. La final setam patrol.robot_hdg=desired.

    patrol.robot_hdg = desired
    set_steering_angle(STEERING_CENTER_ANGLE)

    dist_remaining = math.hypot(patrol.robot_x - wx, patrol.robot_y - wy)
    travel_time2 = dist_remaining / (REAL_SPEED_M_S * (PATROL_DUTY / 100.0))
    print(f"[Patrula] Mers drept: {travel_time2:.1f}s  distanta={dist_remaining:.2f}m")

    DRIVE_DURATION = 5.0
    SCAN_DURATION  = 0.5
    elapsed = 0.0

    while elapsed < travel_time2:
        set_motor_pwm("forward", PATROL_DUTY)
        drive_t0 = time.monotonic()
        while time.monotonic() - drive_t0 < DRIVE_DURATION and elapsed < travel_time2:
            if sonar_blocked():
                set_motor_pwm("stop", 0.0)
                print(f"[Patrula] Obstacol, dau cu spatele virat 50cm")
                if not hasattr(drive_to_waypoint, "_stuck_dir"):
                    drive_to_waypoint._stuck_dir = 1
                drive_to_waypoint._stuck_dir *= -1
                if drive_to_waypoint._stuck_dir > 0:
                    set_steering_angle(STEERING_LEFT_ANGLE + 5)
                else:
                    set_steering_angle(STEERING_RIGHT_ANGLE - 5)
                _reverse_50cm()
                print("[Patrula] Spatele gata, repornesc virat opus")
                if drive_to_waypoint._stuck_dir > 0:
                    set_steering_angle(STEERING_RIGHT_ANGLE - 5)
                else:
                    set_steering_angle(STEERING_LEFT_ANGLE + 5)
                set_motor_pwm("forward", PATROL_DUTY)
                time.sleep(2.0)
                set_steering_angle(STEERING_CENTER_ANGLE)
                set_motor_pwm("stop", 0.0)
                patrol.robot_y = 0.0
                return []
            if _mpu is not None:
                if not hasattr(drive_to_waypoint, '_stuck_st'):
                    drive_to_waypoint._stuck_st = {'suspect_since': None, 'last_active_test': 0.0}
                if check_stuck(drive_to_waypoint._stuck_st, suspect_threshold=4.0):
                    print("[Patrula] Robot blocat confirmat! Dau cu spatele virat")
                    set_motor_pwm("stop", 0.0)
                    time.sleep(0.3)
                    if not hasattr(drive_to_waypoint, '_stuck_dir'):
                        drive_to_waypoint._stuck_dir = 1
                    drive_to_waypoint._stuck_dir *= -1
                    if drive_to_waypoint._stuck_dir > 0:
                        set_steering_angle(STEERING_LEFT_ANGLE + 5)
                    else:
                        set_steering_angle(STEERING_RIGHT_ANGLE - 5)
                    print("[Patrula] Mers inapoi 50cm...")
                    _reverse_50cm()
                    print("[Patrula] Deblocat, repornesc virat in directie opusa")
                    if drive_to_waypoint._stuck_dir > 0:
                        set_steering_angle(STEERING_RIGHT_ANGLE - 5)
                    else:
                        set_steering_angle(STEERING_LEFT_ANGLE + 5)
                    set_motor_pwm("forward", PATROL_DUTY)
                    time.sleep(1.5)
                    set_steering_angle(STEERING_CENTER_ANGLE)
                    set_motor_pwm("stop", 0.0)
                    patrol.robot_y = 0.0
                    return []
            time.sleep(0.03)
            elapsed += 0.03
            patrol.update_pos(STEERING_CENTER_ANGLE, 0.03, PATROL_DUTY)

        if elapsed >= travel_time2:
            break

        set_motor_pwm("stop", 0.0)
        time.sleep(0.15)
        balls = camera_scan_arc(picam2, model, sports_ball_cls_id, fx, fy, cx0, cy0, patrol=patrol)
        if balls:
            print(f"[Patrula] {len(balls)} minge(i) confirmate de YOLO, colectez")
            return balls
        frame_bgr = capture_corrected(picam2)
        state_set(frame_jpg=encode_frame_jpg(annotate_frame(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), [])))
        scan_remaining = SCAN_DURATION - 0.15
        if scan_remaining > 0:
            time.sleep(min(scan_remaining, 0.3))

    set_motor_pwm("stop", 0.0)
    return []


_mpu    = None
_gz_off = 0.0

def read_gz() -> float:
    if _mpu is None: return 0.0
    try:
        return _mpu.get_gyro_data()["z"] - _gz_off
    except Exception:
        return 0.0


def main():
    global _mpu, _gz_off
    
    # RTC: sincronizare ceas sistem + initializare jurnal
    rtc_module.sync_system_clock_from_rtc()
    rtc_module.journal_init()
    rtc_module.log_event("START", "Robot pornit")
    session_start_t = time.monotonic()
    
    init_actuators()
    picam2 = init_camera()

    # Stream live continuu in fundal (independent de YOLO)
    threading.Thread(target=_continuous_stream_thread, args=(picam2,), daemon=True).start()
    print("[Stream] Thread continuu pornit, dashboard live activ")

    try:
        _mpu = MPU6050Class(MPU_ADDR)
        _mpu.set_gyro_range(MPU6050Class.GYRO_RANGE_500DEG)
        print("[MPU6050] Calibrare giroscop 2s...")
        for _ in range(40):
            _gz_off += _mpu.get_gyro_data()["z"]
            time.sleep(0.03)
        _gz_off /= 40
        print(f"[MPU6050] OK. gz_off={_gz_off:.3f} deg/s")
    except Exception as e:
        print(f"[MPU6050] Eroare: {e}, giroscop dezactivat")
        _mpu = None

    # Model custom antrenat (mAP@50=88.1%, cls_id=0)
    MODEL_PATH = "/home/pi/best.pt"
    try:
        model = YOLO(MODEL_PATH)
        print(f"[YOLO] Model custom incarcat: {MODEL_PATH}")
        sports_ball_cls_id = 0
    except Exception as e:
        print(f"[YOLO] EROARE incarcare {MODEL_PATH}: {e}")
        print("[YOLO] Fallback la yolov8s.pt standard (cls_id=32)")
        model = YOLO("yolov8s.pt")
        sports_ball_cls_id = 32
    print("[YOLO] Warmup...")
    _dummy = np.zeros((240, 320, 3), dtype=np.uint8)
    model(_dummy, imgsz=IMG_SIZE, conf=CONFIDENCE_THRESHOLD, verbose=False)
    print(f"[YOLO] Warmup complet. cls_id pentru minge = {sports_ball_cls_id}")

    fx  = FOCAL_LENGTH_PX
    fy  = FOCAL_LENGTH_PX
    cy0 = 480 / 2.0   # 240 pentru 640x480
    cx0 = 309.0     # centrul mecanic al colectorului (calibrat empiric)
    print(f"[Camera] cx0={cx0:.1f}px  (centrul mecanic al colectorului)")

    balls_collected_total = 0
    patrol = PatrolState()
    start_sonar()
    start_dashboard()
    if PREVIEW_ENABLED:
        preview_t = threading.Thread(target=preview_loop, args=(picam2,), daemon=True)
        preview_t.start()
        print(f"[Previzualizare] Fereastra live '{PREVIEW_WINDOW}' deschisa.")

    def collect_balls(balls_found, patrol_anchor_x=None, patrol_anchor_y=None):
        """Colectarea tuturor mingilor detectate, in ordine nearest-neighbour.

        Pentru fiecare minge:
          1. Daca needs_reverse (detectata lateral) - marsarier scurt + rescan
          2. Pre-orientare: verifica daca robotul poate ajunge prin viraj direct
             sau trebuie sa dea cu spatele (dead zone, eroare > 35deg + dist < 1.2m)
          3. Viraj giroscop (target = err * TURN_FACTOR * 0.80) sau timp (backup)
          4. Mers drept calculat din distanta - distanta_parcursa_in_viraj
          5. La final: stop + scan rotativ pentru mingi noi
        """
        nonlocal balls_collected_total
        targets = group_balls_into_targets(balls_found)
        path    = nearest_neighbour_path(targets, start_cx=cx0)
        log_scan_results(balls_found, targets, path)
        state_set(phase="collecting")
        n_run = 0

        for target in path:
            target_cx  = target["cx"]
            distance_z = target["Z"]
            kind       = target["kind"]

            # FLAG needs_reverse: minge detectata cu camera laterala
            # Robotul trebuie sa dea cu spatele pana vede minge la centru
            if target.get("needs_reverse", False):
                direction = target.get("reverse_direction", "DREAPTA")
                print(f"[Pre-orientare] Minge detectata la {direction}, "
                      f"dau cu spatele 30cm")
                # Vireaza in sens opus, la marsarier spatele merge invers
                if direction == "STANGA":
                    set_steering_angle(STEERING_SAFE_RIGHT)
                else:
                    set_steering_angle(STEERING_SAFE_LEFT)
                time.sleep(0.3)
                set_motor_pwm("backward", 70.0)

                BACK_SPEED_MS = 0.10 * 0.70
                TARGET_BACK_DIST = 0.30   # 30cm parcurs in marsarier (minge laterala)
                TARGET_BACK_TIME = TARGET_BACK_DIST / BACK_SPEED_MS
                back_t0 = time.monotonic()
                while time.monotonic() - back_t0 < TARGET_BACK_TIME:
                    time.sleep(0.05)

                set_motor_pwm("stop", 0.0)
                stop_and_center()
                time.sleep(0.5)
                elapsed = time.monotonic() - back_t0
                print(f"[Pre-orientare] Marsarier complet: {elapsed:.1f}s "
                      f"(~{elapsed*BACK_SPEED_MS:.2f}m)")

                # Rescan complet cu rotire camera dupa marsarier
                print("[Pre-orientare] Rescan complet dupa marsarier...")
                new_balls = camera_scan_arc(picam2, model, sports_ball_cls_id,
                                             fx, fy, cx0, cy0, None)
                if new_balls:
                    print(f"[Pre-orientare] {len(new_balls)} minge(i) gasite "
                          f"-- recalculez traseu")
                    return collect_balls(new_balls, patrol_anchor_x, patrol_anchor_y)
                else:
                    print("[Pre-orientare] Rescan: nicio minge, abandonez")
                    continue

            if distance_z <= STOP_DISTANCE_M:
                # Daca mingea e prea laterala chiar daca e aproape,
                # nu putem ajunge direct, lasa branch-ul de marsarier sa o rezolve
                target_x_close = target.get("X_world",
                                  target["balls"][0].get("X_world", 0.0)
                                  if target.get("balls") else 0.0)
                if abs(target_x_close) > 0.15:
                    print(f"[{kind.upper()}] Minge aproape ({distance_z:.2f}m) "
                          f"dar lateral={target_x_close:+.2f}m, nu pot direct, "
                          f"continui la branch marsarier")
                    # NU continue, las codul sa continue la verificarea marsarier
                else:
                    # Verificare YOLO inainte de colectare, mingile foarte aproape
                    # sunt suspecte (poate detectie falsa pe podea/reflexie)
                    print(f"[{kind.upper()}] Minge foarte aproape ({distance_z:.3f}m), verific cu YOLO")
                    stop_and_center()
                    time.sleep(0.2)
                    verify_frame = capture_corrected(picam2)
                    verify_results = model(verify_frame, imgsz=IMG_SIZE,
                                            conf=CONFIDENCE_THRESHOLD, verbose=False)
                    ball_still_visible = False
                    for r in verify_results:
                        if r.boxes is None: continue
                        for box in r.boxes:
                            if int(box.cls[0]) == sports_ball_cls_id:
                                ball_still_visible = True
                                break
                        if ball_still_visible: break

                    if not ball_still_visible:
                        print(f"[{kind.upper()}] YOLO nu confirma minge, detectie falsa, sar")
                        continue

                    print(f"[{kind.upper()}] Confirmata YOLO, merg 5.0s")
                    set_steering_angle(pid.compute(target_cx, 640.0))
                    time.sleep(0.2)
                    set_motor_pwm("forward", COLLECT_DUTY)
                    time.sleep(5.0)
                    set_motor_pwm("stop", 0.0)
                    n = len(target["balls"])
                    balls_collected_total += n;  n_run += n
                    state_set(balls_collected=balls_collected_total)
                    print(f"Colectat {n} minge(i). Total: {balls_collected_total}")
                    rtc_module.log_event("BALL_COLLECT",
                        f"{n} minge(i) colectate (apropiat). Total: {balls_collected_total}")

                    # Scan rotativ dupa fiecare colectare, cauta mingi noi
                    print("[Colectare] Scan rotativ pentru mingi noi...")
                    stop_and_center()
                    time.sleep(0.3)
                    new_balls = camera_scan_arc(picam2, model, sports_ball_cls_id,
                                                 fx, fy, cx0, cy0, None)
                    if new_balls:
                        print(f"[Colectare] {len(new_balls)} minge(i) gasite, recalculez traseu")
                        return collect_balls(new_balls, patrol_anchor_x, patrol_anchor_y)
                    else:
                        print("[Colectare] Nicio minge noua, continuu cu lista existenta")
                    continue

            target_lateral = target.get("balls", [{}])[0].get("X_world", 0.0)
            in_collector_range = abs(target_lateral) <= (COLLECTOR_DIAMETER_M + 0.02)
            print(f"[{kind.upper()}] Urmarire vizuala: cx={target_cx:.1f}  "
                  f"adancime={distance_z:.3f}m  lateral={target_lateral:+.3f}m  "
                  f"in_colector={in_collector_range}")

            pid.reset(seed_x=target_cx, frame_width=640.0)
            init_err = (target_cx - cx0) / cx0
            init_angle = pid.compute(target_cx, 640.0)

            target_x_world_pre = target.get("X_world",
                                 target["balls"][0].get("X_world", 0.0)
                                 if target.get("balls") else 0.0)
            _pre_bgr = capture_corrected(picam2)
            _already_visible = has_ball_color(_pre_bgr)
            _dist_pre = target.get("Z", distance_z)
            _cam_was_rotated = abs(target.get("cam_angle", CAMERA_CENTER) - CAMERA_CENTER) > 10.0
            # Unghi real al mingii in lume (nu degrees pe numarul normalizat de pixeli)
            _err_deg = abs(math.degrees(math.atan2(target_x_world_pre, max(_dist_pre, 0.01))))

            # Zona moarta geometrica: tinta inaccesibila prin viraj direct
            # daca e in interiorul cercului de viraj minim.
            R_VIRAJ_EFF = 0.45   # raza viraj efectiva (relaxat)
            _x_lat  = abs(target_x_world_pre)
            _depth  = _dist_pre
            _in_dead_zone = (_x_lat**2 + _depth**2) < (2.0 * R_VIRAJ_EFF * _x_lat)

            _need_reverse_soft = (_cam_was_rotated and _dist_pre < 0.70)

            # STRICT, robotul fizic NU poate ajunge prin viraj direct.
            _need_reverse_strict = _in_dead_zone or (
                _err_deg > 90.0 and _dist_pre < 1.00
            ) or (
                _dist_pre < 0.40 and _x_lat > 0.30
            ) or (
                # Mingea aproape lateral cat adancimea, prea agresiv pentru viraj
                _err_deg > 35.0 and _dist_pre < 1.20
            ) or (
                # Minge aproape si laterala (err moderat, dar dist mica)
                _err_deg > 20.0 and _dist_pre < 0.80
            )

            _need_reverse = _need_reverse_soft or _need_reverse_strict
            print(f"[Pre-orientare] decisie: strict={_need_reverse_strict} "
                  f"soft={_need_reverse_soft} vizibil={_already_visible} "
                  f"- marsarier={_need_reverse_strict or (_need_reverse_soft and not _already_visible)}")

            _should_reverse = _need_reverse_strict or (_need_reverse_soft and not _already_visible)
            if _should_reverse:
                print(f"[Pre-orientare] Minge inaccesibila prin viraj direct "
                      f"(err={init_err:+.2f}, lat={target_x_world_pre:+.2f}m, "
                      f"z={_dist_pre:.2f}m), dau cu spatele")
                if init_err < 0:
                    # minge stanga - spatele sa mearga spre stanga - servo dreapta
                    set_steering_angle(STEERING_RIGHT_ANGLE - 5)
                else:
                    # minge dreapta - spatele sa mearga spre dreapta - servo stanga
                    set_steering_angle(STEERING_LEFT_ANGLE + 5)
                time.sleep(0.3)
                set_motor_pwm("backward", 70.0)

                MIN_BACK_DIST_M = 0.50 if _need_reverse_strict else 0.0
                BACK_SPEED_MS = 0.10 * 0.70
                MAX_BACK = max(9.0, (MIN_BACK_DIST_M / BACK_SPEED_MS) + 2.0)
                back_t0  = time.monotonic()
                found_color = False
                while time.monotonic() - back_t0 < MAX_BACK:
                    elapsed = time.monotonic() - back_t0
                    dist_back = elapsed * BACK_SPEED_MS
                    f_bgr = capture_corrected(picam2)
                    if dist_back < MIN_BACK_DIST_M:
                        time.sleep(0.03)
                        continue
                    if has_ball_color(f_bgr):
                        print(f"[Pre-orientare] Culoare minge detectata la "
                              f"t={elapsed:.1f}s dist={dist_back:.2f}m, opresc")
                        found_color = True
                        break
                    time.sleep(0.03)

                set_motor_pwm("stop", 0.0)
                time.sleep(0.5)   # pauza stabilizare imagine
                if not found_color and _need_reverse_strict:
                    elapsed = time.monotonic() - back_t0
                    dist_back = elapsed * BACK_SPEED_MS
                    if dist_back >= MIN_BACK_DIST_M:
                        print(f"[Pre-orientare] Marsarier strict complet "
                              f"({dist_back:.2f}m), scanez")
                        found_color = True

                if found_color:
                    # Scan SIMPLU cu camera centrata (culoarea verde e deja detectata)
                    print("[Pre-orientare] Oprit dupa marsarier, YOLO frontal pentru coordonate")
                    stop_and_center()
                    time.sleep(0.3)
                    f_bgr = capture_corrected(picam2)
                    yolo_res = model(f_bgr, imgsz=IMG_SIZE, conf=CONFIDENCE_THRESHOLD, verbose=False)
                    new_balls = []
                    for r in yolo_res:
                        if r.boxes is None: continue
                        for box in r.boxes:
                            if int(box.cls[0]) != sports_ball_cls_id: continue
                            x1,y1,x2,y2 = map(int, box.xyxy[0])
                            cxb = (x1+x2)//2
                            wpx = x2-x1
                            p3d = compute_3d_position(cxb,(y1+y2)//2,wpx,fx,fy,cx0,cy0)
                            if p3d is None: continue
                            X,Y,Z = p3d
                            if not _is_plausible_ball(X, Z):
                                continue
                            new_balls.append({"cx":cxb,"cy":(y1+y2)//2,"Z":Z,"X_world":X,
                                              "box":(x1,y1,x2,y2),"cam_angle": CAMERA_CENTER})
                    if new_balls:
                        print(f"[Pre-orientare] YOLO frontal: {len(new_balls)} minge(i), recalculez traseu")
                        return collect_balls(new_balls, patrol_anchor_x, patrol_anchor_y)
                    else:
                        print("[Pre-orientare] YOLO frontal: nicio minge, target abandonat")
                        stop_and_center()
                        continue
                else:
                    print("[Pre-orientare] Marsarier fara detectie culoare, target abandonat")
                    stop_and_center()
                    continue

            # Mingea NU necesita marsarier, abordare directa
            pid.reset(seed_x=target_cx, frame_width=640.0)
            init_angle = pid.compute(target_cx, 640.0)
            print(f"[Pre-orientare] Abordare directa: servo={init_angle:.1f}deg")
            for _ in range(3):
                set_steering_angle(init_angle)
                time.sleep(0.2)

            # ===== NAVIGARE spre minge, logica identica cu drive_to() din test_navigare.py =====
            # Coordonatele tintei in sistemul de referinta al robotului:
            #   wx_ball = X_world (lateral, stanga negativ, dreapta pozitiv)
            #   wy_ball = distance_z (inainte, mereu pozitiv)
            target_x_world = target.get("X_world",
                             target["balls"][0].get("X_world", 0.0)
                             if target.get("balls") else 0.0)
            if kind == "pair" and target.get("balls"):
                balls_x = [b.get("X_world", 0.0) for b in target["balls"]]
                centru_x = sum(balls_x) / len(balls_x)
                laterala_x = max(balls_x, key=abs)
                semn = 1.0 if laterala_x > 0 else -1.0
                target_x_world = centru_x + semn * 0.05

            # Pozitia tintei relativa la robot (robot la origine, heading=0 = inainte)
            wx_ball = target_x_world
            wy_ball = distance_z

            # Unghi dorit si eroare fata de heading curent (0.0 la inceputul fiecarei colectari)
            ball_hdg  = 0.0   # heading local reset la fiecare colectare
            desired_b = math.atan2(wx_ball, wy_ball)
            err_b     = (desired_b - ball_hdg + math.pi) % (2 * math.pi) - math.pi
            err_deg_b = math.degrees(err_b)
            norm_b    = clamp(err_b / math.radians(30), -1.0, 1.0)   # 30deg = maxim (ca in test_navigare)

            if norm_b >= 0:
                servo_b = STEERING_CENTER_ANGLE + norm_b * (STEERING_RIGHT_ANGLE - STEERING_CENTER_ANGLE)
            else:
                servo_b = STEERING_CENTER_ANGLE + norm_b * (STEERING_CENTER_ANGLE - STEERING_LEFT_ANGLE)

            dist_blind = math.hypot(wx_ball, wy_ball)
            print(f"[Nav] wx={wx_ball:+.2f}m  wy={wy_ball:.2f}m  "
                  f"desired={math.degrees(desired_b):.1f}deg  "
                  f"err={err_deg_b:.1f}deg  servo={servo_b:.1f}deg  dist={dist_blind:.2f}m")
            state_set(phase="collecting")

            min_err = 0.0 if kind == "pair" else 5.0


            # --- Pas 1: viraj cu giroscop ---
            # Dead reckoning local: robot porneste de la (loc_x=0, loc_y=0)
            # cu heading=0 (inainte). Urmarim pozitia in timpul virajului
            # exact ca in test_navigare (update_pos in bucla).
            loc_x   = 0.0
            loc_y   = 0.0
            ball_hdg = 0.0   # heading local, reset la fiecare colectare

            if abs(err_deg_b) > min_err:
                if err_deg_b < 0:
                    dir_str          = "STANGA"
                    dps              = DEGREES_PER_SECOND_LEFT
                    servo_turn       = STEERING_SAFE_LEFT
                    gz_sign_expected = +1.0
                else:
                    dir_str          = "DREAPTA"
                    dps              = DEGREES_PER_SECOND_RIGHT
                    servo_turn       = STEERING_SAFE_RIGHT
                    gz_sign_expected = -1.0

                if USE_TIME_BASED_TURN:
                    # ---- Viraj pe TIMP ----
                    turn_time = abs(err_deg_b) / dps
                    print(f"[Nav] Viraj timp {dir_str}: servo={servo_turn:.1f}deg  "
                          f"grade={abs(err_deg_b):.1f}  timp={turn_time:.2f}s  "
                          f"({dps:.2f}deg/s)")
                    set_steering_angle(servo_turn)
                    time.sleep(0.3)
                    set_motor_pwm("forward", 80.0)
                    t0_turn_b = time.monotonic()
                    while time.monotonic() - t0_turn_b < turn_time:
                        if sonar_blocked():
                            break
                        time.sleep(0.02)
                    set_motor_pwm("stop", 0.0)
                    time.sleep(0.2)
                    elapsed_turn = min(time.monotonic() - t0_turn_b, turn_time)
                    print(f"[Nav] Viraj complet: {elapsed_turn:.2f}s")
                else:
                    # ---- Viraj pe GIROSCOP ----
                    target_turn = abs(err_deg_b) * 0.80   # vireaza 80% din eroare
                    print(f"[Nav] Viraj giroscop {dir_str}: servo={servo_turn:.1f}deg  "
                          f"err={abs(err_deg_b):.1f}deg  tinta={target_turn:.1f}deg")
                    set_steering_angle(servo_turn)
                    time.sleep(0.3)
                    set_motor_pwm("forward", 80.0)
                    yaw_net   = 0.0
                    last_t_gb = time.monotonic()
                    t0_turn_b = time.monotonic()
                    while abs(yaw_net) < target_turn:
                        if sonar_blocked(): break
                        if time.monotonic() - t0_turn_b > 15.0:
                            print("[Nav] Timeout viraj giroscop")
                            break
                        now_gb    = time.monotonic()
                        dt_gb     = now_gb - last_t_gb
                        last_t_gb = now_gb
                        gz = read_gz()
                        gz_signed = gz * gz_sign_expected
                        if abs(gz_signed) > 1.0:
                            yaw_net += gz_signed * dt_gb
                        time.sleep(0.02)
                    set_motor_pwm("stop", 0.0)
                    time.sleep(0.2)
                    elapsed_turn = time.monotonic() - t0_turn_b
                    print(f"[Nav] Viraj giroscop complet: yaw_net={abs(yaw_net):.1f}/{target_turn:.1f}deg "
                          f"({elapsed_turn:.2f}s)")

                # Dead reckoning pozitie dupa viraj (acelasi pentru ambele)
                speed_turn = REAL_SPEED_M_S * (80.0 / 100.0)
                norm_t = (servo_turn - STEERING_CENTER_ANGLE) / (STEERING_RIGHT_ANGLE - STEERING_CENTER_ANGLE)
                ball_hdg += norm_t * math.radians(30) * elapsed_turn
                loc_x    += speed_turn * math.sin(ball_hdg) * elapsed_turn
                loc_y    += speed_turn * math.cos(ball_hdg) * elapsed_turn
                ball_hdg = desired_b
            else:
                print(f"[Nav] Eroare mica ({err_deg_b:.1f}deg), merg drept")

            # --- Pas 2: mers drept spre minge ---
            # Folosim dist_blind original, dead reckoning dupa viraj e imprecis
            # Compensare: in timpul virajului robotul a parcurs deja o distanta
            set_steering_angle(STEERING_CENTER_ANGLE)
            try:
                dist_in_turn = REAL_SPEED_M_S * (80.0 / 100.0) * elapsed_turn
            except NameError:
                dist_in_turn = 0.0
            dist_remaining = max(0.0, dist_blind - dist_in_turn)
            travel_calc  = dist_remaining / REAL_SPEED_M_S
            travel_blind = travel_calc + COLLECT_EXTRA_TIME_S
            print(f"[Nav] Mers drept: {travel_blind:.1f}s "
                  f"(calculat={travel_calc:.1f}s + extra={COLLECT_EXTRA_TIME_S:.1f}s) "
                  f"distanta_ramasa={dist_remaining:.2f}m "
                  f"(parcurs in viraj={dist_in_turn:.2f}m)")
            set_motor_pwm("forward", PATROL_DUTY)
            t0_blind = time.monotonic()
            MAX_BLIND_TIME = travel_blind * 1.5 + 2.0
            collection_failed_reason = None
            while time.monotonic() - t0_blind < travel_blind:
                if sonar_blocked():
                    collection_failed_reason = "sonar (obstacol detectat inainte de a ajunge la minge)"
                    break
                if time.monotonic() - t0_blind > MAX_BLIND_TIME:
                    print("[Nav] Timeout absolut, oprire")
                    collection_failed_reason = "timeout absolut"
                    break
                if _mpu is not None:
                    if not hasattr(collect_balls, '_stuck_st_blind'):
                        collect_balls._stuck_st_blind = {'suspect_since': None, 'last_active_test': 0.0}
                    t_before_stuck = time.monotonic()
                    is_stuck = check_stuck(collect_balls._stuck_st_blind, suspect_threshold=10.0)
                    t_stuck_elapsed = time.monotonic() - t_before_stuck
                    # Testul de stuck consuma timp in care robotul NU avanseaza spre minge.
                    # Compensam: impingem t0_blind inainte cu timpul consumat, ca sa nu
                    # pierdem din timpul de mers real (altfel se opreste inainte de minge).
                    if t_stuck_elapsed > 0.1:
                        t0_blind += t_stuck_elapsed
                        set_motor_pwm("forward", PATROL_DUTY)  # reia mersul dupa test
                    if is_stuck:
                        print("[Nav] Robot blocat in colectare, dau cu spatele")
                        set_motor_pwm("stop", 0.0)
                        time.sleep(0.2)
                        if not hasattr(drive_to_waypoint, "_stuck_dir"):
                            drive_to_waypoint._stuck_dir = 1
                        drive_to_waypoint._stuck_dir *= -1
                        if drive_to_waypoint._stuck_dir > 0:
                            set_steering_angle(STEERING_LEFT_ANGLE + 5)
                        else:
                            set_steering_angle(STEERING_RIGHT_ANGLE - 5)
                        _reverse_50cm()
                        set_steering_angle(STEERING_CENTER_ANGLE)
                        collection_failed_reason = "blocaj mecanic (stuck)"
                        break
                time.sleep(0.03)
            else:
                collection_failed_reason = None  # bucla s-a terminat normal (distanta parcursa complet)

            set_motor_pwm("stop", 0.0)
            if collection_failed_reason:
                print(f"[Nav] Colectare NECONFIRMATA - oprire prematura: {collection_failed_reason}")
                print(f"[Nav] Minge NU este marcata ca si colectata")
            else:
                print("[Nav] Colectare finalizata")

            set_motor_pwm("stop", 0.0)
            if not collection_failed_reason:
                n_col = len(target["balls"])
                balls_collected_total += n_col;  n_run += n_col
                state_set(balls_collected=balls_collected_total)
                update_status_led(balls_collected_total)
                print(f"Colectat {n_col}. Total: {balls_collected_total}")
                rtc_module.log_event("BALL_COLLECT",
                    f"{n_col} minge(i) colectate (mers drept). Total: {balls_collected_total}")
            else:
                n_col = 0
                print(f"Colectare esuata, minge neconfirmata. Total ramane: {balls_collected_total}")

            if balls_collected_total >= CAPACITY_MAX:
                print("\n" + "="*60)
                print(f"  [MAIN] COLECTOR PLIN ({balls_collected_total} mingi)")
                print(f"  [MAIN] Pornesc intoarcerea la statia de baza")
                print("="*60)
                stop_and_center()
                time.sleep(0.5)
                hw = {
                    "set_motor_pwm":          set_motor_pwm,
                    "set_steering_angle":     set_steering_angle,
                    "stop_and_center":        stop_and_center,
                    "sonar_blocked":          sonar_blocked,
                    "sonar_m_now":            lambda: state_get("sonar_m"),
                    "check_stuck":            check_stuck,
                    "state_set":              state_set,
                    "STEERING_CENTER_ANGLE":  STEERING_CENTER_ANGLE,
                    "STEERING_LEFT_ANGLE":    STEERING_SAFE_LEFT,
                    "STEERING_RIGHT_ANGLE":   STEERING_SAFE_RIGHT,
                }
                state_set(phase="homing")
                rtc_module.log_event("HOMING_START",
                    f"Colector plin ({balls_collected_total} mingi), start homing")
                homing_ok = home_to_station(picam2, hw, wait_after_arrive=True)
                if homing_ok:
                    rtc_module.log_event("HOMING_ARRIVED", "Ajuns la statie, colector golit")
                    balls_collected_total = 0
                    n_run                 = 0
                    state_set(balls_collected=0, phase="patrol")
                    update_status_led(0)
                    print("[MAIN] Colector golit. Reluare patrulare.\n")
                    return n_run
                else:
                    rtc_module.log_event("ERROR", "Homing esuat")
                    print("[MAIN] HOMING ESUAT, continui cu colector plin")
                    state_set(phase="patrol")
                    return n_run

            stop_and_center()
            time.sleep(0.3)

            # Scan rotativ dupa fiecare colectare, cauta mingi noi
            print("[Colectare] Scan rotativ pentru mingi noi...")
            new_balls = camera_scan_arc(picam2, model, sports_ball_cls_id,
                                         fx, fy, cx0, cy0, None)
            if new_balls:
                print(f"[Colectare] {len(new_balls)} minge(i) gasite, recalculez traseu")
                return collect_balls(new_balls, patrol_anchor_x, patrol_anchor_y)
            else:
                print("[Colectare] Nicio minge noua, continuu cu lista existenta")

        # === SFARSIT path, toate mingile din scanul initial au fost colectate ===
        # Nu mai facem scan suplimentar aici - ultimul scan din bucla
        # (dupa ultima minge colectata) deja a verificat zona.
        stop_and_center()

        stop_and_center()
        if patrol_anchor_x is not None:
            lateral_drift = patrol.robot_x - patrol_anchor_x
            print(f"[Revenire] Abatere laterala estimata: {lateral_drift:+.3f}m")
            # Eroare heading: dorim sa fim aliniati cu axa Y (hdg=0)
            hdg_err_deg = -math.degrees(patrol.robot_hdg)
            if abs(hdg_err_deg) > 5.0:
                norm_ret = clamp(hdg_err_deg / 30.0, -1.0, 1.0)
                if norm_ret >= 0:
                    angle_ret = STEERING_CENTER_ANGLE + norm_ret * (STEERING_RIGHT_ANGLE - STEERING_CENTER_ANGLE)
                else:
                    angle_ret = STEERING_CENTER_ANGLE + norm_ret * (STEERING_CENTER_ANGLE - STEERING_LEFT_ANGLE)
                print(f"[Revenire] Intoarcere {hdg_err_deg:.1f}deg servo={angle_ret:.1f}deg")
                set_steering_angle(angle_ret)
                time.sleep(0.2)
                turn_t = clamp(abs(hdg_err_deg) * 0.5 / 20.0, 0.3, 4.0)
                set_motor_pwm("forward", TURN_SPEED_DUTY)
                time.sleep(turn_t)
                set_motor_pwm("stop", 0.0)
                patrol.robot_hdg = 0.0
                set_steering_angle(STEERING_CENTER_ANGLE)
                print(f"[Revenire] Realiniat in {turn_t:.1f}s")
                set_motor_pwm("forward", PATROL_DUTY)
                time.sleep(0.5)
                set_motor_pwm("stop", 0.0)
            else:
                print(f"[Revenire] Servo centrat. Patrularea continua.")
        return n_run

    try:
        while True:
            waypoints = build_patrol_grid()
            patrol.reset(waypoints)
            recorder.clear()
            stop_and_center()
            print("\n" + "="*60)
            print(f"  START PATRULA, {len(waypoints)} puncte de control")
            print("="*60)

            state_set(phase="scanning")
            pid.reset()
            stop_and_center()
            print("\n[Scanare] Scanare initiala cu rotatie camera...")
            initial_balls = camera_scan_arc(picam2, model, sports_ball_cls_id, fx, fy, cx0, cy0, patrol=patrol)
            print(f"[Scanare] Final: {len(initial_balls)} minge(i) detectata(e)")
            if initial_balls:
                collect_balls(initial_balls)
                recorder.clear()
            else:
                print("[Scanare] Nicio minge detectata. Pornesc patrularea.")

            while not patrol.all_visited():
                result = patrol.next_in_sequence()
                if result is None:
                    break
                wp_idx, (wx, wy) = result
                print(f"\n[Patrula] {patrol.progress()}")
                print(f"[Patrula] - PC {wp_idx}: ({wx:.2f},{wy:.2f})m")
                found = drive_to_waypoint(patrol, wx, wy, picam2, model,
                                          sports_ball_cls_id, fx, fy, cx0, cy0)
                if found:
                    anchor_x = patrol.robot_x
                    anchor_y = patrol.robot_y
                    print(f"[Anchor] Pozitie patrula salvata: x={anchor_x:.3f}m, y={anchor_y:.3f}m")
                    collect_balls(found, patrol_anchor_x=anchor_x, patrol_anchor_y=anchor_y)
                    recorder.clear()

                    print("[Patrula] Verific cu YOLO inainte de reluare...")
                    time.sleep(0.3)
                    f_bgr = capture_corrected(picam2)
                    res_check = model(f_bgr, imgsz=IMG_SIZE, conf=CONFIDENCE_THRESHOLD, verbose=False)
                    obstacole_found = []
                    for r in res_check:
                        if r.boxes is None: continue
                        for box in r.boxes:
                            x1,y1,x2,y2 = map(int, box.xyxy[0])
                            wpx = x2-x1
                            cxb = (x1+x2)//2
                            cls_id = int(box.cls[0])
                            p3d = compute_3d_position(cxb,(y1+y2)//2,wpx,fx,fy,cx0,cy0)
                            if p3d is None: continue
                            X,Y,Z = p3d
                            if cls_id == sports_ball_cls_id:
                                print(f"[Patrula] YOLO: minge la {Z:.2f}m lateral={X:+.2f}m")
                                obstacole_found.append({"cx":cxb,"cy":(y1+y2)//2,"Z":Z,"X_world":X,"box":(x1,y1,x2,y2)})
                    if obstacole_found:
                        print(f"[Patrula] {len(obstacole_found)} minge(i) detectate, colectez")
                        collect_balls(obstacole_found, patrol_anchor_x=anchor_x, patrol_anchor_y=anchor_y)
                    else:
                        print("[Patrula] Drum liber, reiau patrularea")
                    print("[Patrula] Mingi colectate. Reiau patrularea.")
                    patrol.mark_visited(wp_idx)
                else:
                    patrol.mark_visited(wp_idx)

            print("\n" + "="*60)
            print(f"  PATRULA COMPLETA, {balls_collected_total} minge(i) colectata(e)")
            print("="*60)
            state_set(phase="idle")
            print("Astept 5s inainte de urmatoarea runda...")
            time.sleep(5.0)

    except KeyboardInterrupt:
        print("Oprire utilizator.")
        rtc_module.log_event("STOP", "Oprire de la utilizator (Ctrl+C)")
        try:
            set_motor_pwm("stop", 0.0)
            set_steering_angle(STEERING_CENTER_ANGLE)
        except:
            pass
    finally:
        # Sumar sesiune
        try:
            duration = time.monotonic() - session_start_t
            rtc_module.log_session_summary(balls_collected_total, duration, success=True)
        except Exception:
            pass
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        cleanup_actuators()


if __name__ == "__main__":
    main()
