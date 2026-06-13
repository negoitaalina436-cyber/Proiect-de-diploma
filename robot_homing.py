"""
robot_homing.py -- modul de intoarcere la statia de baza prin MQTT + RSSI WiFi
                   + ghidare vizuala prin LED-uri verzi.

Folosire ca modul (din robot.py):
    from robot_homing import home_to_station
    success = home_to_station(picam2, hw)

Folosire standalone (test):
    python3 robot_homing.py

Flux:
  1. Publica "INTOARCERE" pe MQTT broker.hivemq.com (topic licenta_robot/status)
  2. ESP32 statie aprinde 4 LED-uri verzi + activeaza AP Far_Statie_Baza
  3. Robotul scaneaza WiFi prin iwlist si navigheaza prin gradient RSSI (FAR)
  4. Cand RSSI > -70 dBm trece la NEAR si cauta vizual LED-urile verzi
  5. La detectare blob mare verde -- opreste, confirma static (>=2/5 detectii)
  6. Calculeaza Z, X, err -- vireaza spre statie, merge orbeste pana la 60cm
  7. Publica "AJUNS" si asteapta 10s pentru descarcare manuala
  8. Publica "GOL" -- statia stinge LED-uri + AP
  9. Dashboard live pe portul 5002 (stream camera + masca HSV)
"""

import time
import subprocess
import re
import threading
import math

import cv2
import numpy as np

try:
    import rtc_module
except ImportError:
    rtc_module = None

# Flask pentru dashboard live pe portul 5002
try:
    from flask import Flask, Response, render_template_string
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Frame-uri partajate pentru stream
_last_frame_bgr = None
_last_mask      = None
_last_info      = "Astept date..."
_frame_lock     = threading.Lock()

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    print("[Homing] AVERTISMENT: paho-mqtt nu e instalat -- "
          "ruleaza: pip install paho-mqtt")
    MQTT_AVAILABLE = False


# ========================================================
# CONFIGURATIE MQTT & WiFi
# ========================================================
BROKER          = "broker.hivemq.com"
PORT            = 1883
TOPIC           = "licenta_robot/status"
CLIENT_ID       = "Robot_Tennis_001"
FAR_WIFI_SSID   = "Far_Statie_Baza"

MSG_START       = "INTOARCERE"   # robotul pleaca spre statie
MSG_ARRIVED     = "AJUNS"        # robotul a ajuns
MSG_UNLOADED    = "GOL"          # colector golit, statia poate stinge LED


# ========================================================
# CONFIGURATIE NAVIGARE -- STATE MACHINE 3 ETAPE
# ========================================================
# Etape de apropiere:
#   FAR         -> doar RSSI WiFi, viteza mare, scop: ajunge in raza statiei
#   NEAR        -> RSSI + scanare blob LED-uri verzi, viteza medie
#                  La confirm static -> calc Z/X/err si drive blind spre statie
#   VISUAL_LOCK -> NEFOLOSIT (legacy -- inlocuit cu drive blind dupa confirm NEAR)

# State names
STATE_FAR         = "FAR"
STATE_NEAR        = "NEAR"
STATE_VISUAL_LOCK = "VISUAL_LOCK"   # legacy, nefolosit in fluxul nou

# Praguri RSSI (dBm) -- pentru tranzitia FAR -> NEAR
RSSI_NEAR_THRESHOLD = -75   # peste aceasta valoare = intram in NEAR
RSSI_DROP_DELTA     = 4     # daca media scade cu >4 dBm -> reorientare
RSSI_HISTORY_SIZE   = 4     # numar citiri pastrate pentru media mobila

# Tranzitii state machine (legacy VL -- nefolosite in flux nou)
LED_DETECT_CONFIRM = 3
LED_LOST_THRESHOLD = 20

# Fallback: RSSI foarte bun fara detectie LED -> tranzitie fortata VL (legacy)
RSSI_FORCE_VL_THRESHOLD = -55   # dBm
RSSI_FORCE_VL_COUNT     = 2     # citiri consecutive ≥ -55 → VISUAL_LOCK fortat

# Detectie blob LED-uri verzi statie
# LED-urile sunt foarte luminoase si satureaza camera -- apar galben-verde pal.
# HSV calibrat empiric: acopera intervalul galben-verde saturat (V>=180).
LED_HSV_LOW         = ( 20,  30, 180)
LED_HSV_HIGH        = ( 90, 255, 255)

# Estimare distanta la statie din bbox-ul blob-ului LED-urilor.
# STATION_WIDTH_M e o valoare "aparenta" -- LED-urile saturate apar mult mai
# mari decat distanta fizica reala intre ele (10cm). Calibrat empiric.
STATION_WIDTH_M       = 0.40      # calibrat empiric (NU e latimea fizica reala)
STATION_FOCAL_LENGTH  = 320.0     # acelasi ca la minge
STATION_STOP_DIST_M   = 0.60      # opreste cand Z <= 60cm
STATION_CX0           = 289.0     # centrul mecanic al colectorului (calibrat)
LED_MIN_AREA_DETECT = 5       # min pixeli per blob (zgomot)
LED_AREA_ARRIVED    = 2000    # NEFOLOSIT (vechi -- folosim Z-distance acum)
LED_SONAR_STOP_M    = 0.50    # NEFOLOSIT (statia e prea joasa pentru sonar)
ORBIT_SONAR_M      = 0.55    # NEFOLOSIT
ORBIT_DURATION_S   = 2.5     # NEFOLOSIT

# Centrul mecanic al colectorului (pentru calcul err pozitie blob)
IMG_CENTER_X       = 289     # calibrat empiric

# Parametri VISUAL_LOCK (cod legacy -- VL inca apelat in fluxul vechi)
LED_STEER_GAIN      = 30.0
LED_STEER_MAX_OFF   = 30.0

# Viteze motor (% duty) per etapa de homing
HOMING_DUTY_FAR        = 90.0   # FAR -- viteza mare
HOMING_DUTY_NEAR_HIGH  = 70.0   # NEAR cand RSSI departe (~ -75 dBm)
HOMING_DUTY_NEAR_LOW   = 50.0   # NEAR cand RSSI aproape (>= -60 dBm)
HOMING_DUTY_VISUAL     = 50.0   # NEFOLOSIT direct (legacy VL)
HOMING_DUTY_TURN       = 65.0   # rotire cautare AP
HOMING_DUTY_REVERSE    = 70.0   # marsarier la deblocare

# Praguri pentru scalarea vitezei in NEAR
NEAR_RSSI_FAST_DBM   = -75   # >= -75 dBm -> incepem sa incetinim
NEAR_RSSI_SLOW_DBM   = -60   # >= -60 dBm -> viteza minima HOMING_DUTY_NEAR_LOW

# Timing
WIFI_SCAN_INTERVAL_S = 2.0
TIMEOUT_S            = 180.0   # 3 minute global
ARRIVE_WAIT_S        = 10.0    # asteptare la baza (descarcare manuala)
ESP32_BOOT_S         = 4.0     # cat dureaza AP-ul sa devina vizibil
ROTATE_SEARCH_S      = 1.5     # rotire cand AP pierdut

# Ocolire obstacol + manevra deblocare (marsarier virat -- pattern din main)
# Calibrat pe baza testului real: la viraj maxim, robotul are raza de
# viraj ~42cm. Pentru o ocolire reala (deplasare laterala ~30cm) e nevoie
# de aprox. 2s de mers virat la 75% duty.
AVOID_BACK_MAX_DIST_M = 1.5    # cat sa dea cu spatele maxim
AVOID_BACK_DUTY       = 70.0   # duty marsarier
AVOID_FWD_DUTY        = 75.0   # duty inainte dupa marsarier
AVOID_SONAR_CLEAR_M   = 1.5    # daca sonar > 1.5m, stop marsarier (e liber)
AVOID_REAL_SPEED_MS   = 0.10   # m/s estimat la 90% duty (din main)
AVOID_FWD_MAX_S       = 4.0    # timp maxim mers inainte virat dupa ocolire
                               # (marit de la 3.0: la raza 42cm e nevoie de
                               #  ~2s pentru o ocolire utila, +margin)
AVOID_FWD_MIN_S       = 2.0    # timp MINIM mers inainte virat dupa ocolire
                               # (chiar daca dist_back e scurt, trebuie sa
                               #  facem o ocolire reala -- altfel revenim
                               #  pe acelasi traseu)

# Unghi reorientare cand RSSI scade
HOMING_STEER_OFFSET_DEG = 20.0


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ========================================================
# Clasa principala
# ========================================================
class HomingController:
    """
    Controlleaza intoarcerea la statie. Primeste hardware-ul deja initializat
    din robot.py (motor, servo, sonar, stuck detection) via dict `hw`.

    `hw` trebuie sa contina:
      - set_motor_pwm(direction, duty)
      - set_steering_angle(angle)
      - stop_and_center()
      - sonar_blocked() -> bool
      - sonar_m_now() -> float
      - check_stuck(state) -> bool
      - state_set(**kwargs)
      - STEERING_CENTER_ANGLE, STEERING_LEFT_ANGLE, STEERING_RIGHT_ANGLE (floats)

    State machine cu 2 etape principale:
      FAR  -- doar RSSI WiFi, viteza mare (90%)
      NEAR -- RSSI + scanare LED-uri verzi, viteza medie (50-70%)
              La detectare blob mare -> confirm static -> calc Z/X/err
              -> viraj giroscop -> drive blind pana la STATION_STOP_DIST_M
    """

    def __init__(self, picam2, hw):
        self.picam2 = picam2
        self.hw     = hw
        self.mqtt_client = None
        # State machine
        self._state = STATE_FAR
        self._consecutive_led_detect = 0   # incrementat in NEAR cand vad LED
        self._consecutive_led_lost   = 0   # incrementat in VISUAL_LOCK cand pierd LED
        self._consecutive_rssi_high = 0   # incrementat in NEAR cand RSSI ≥ -55
        # Istoric RSSI pentru detectia tendintei (media mobila)
        self._rssi_history = []
        # Stuck detection state (refolosit de check_stuck din main)
        self._stuck_state = {"suspect_since": None, "last_active_test": 0.0}
        # Alternare directie ocolire (+1 = stanga, -1 = dreapta)
        self._stuck_dir = 1

    def _push_rssi(self, rssi):
        """Adauga RSSI in istoric, pastreaza ultimele RSSI_HISTORY_SIZE."""
        self._rssi_history.append(rssi)
        if len(self._rssi_history) > RSSI_HISTORY_SIZE:
            self._rssi_history.pop(0)

    def _rssi_is_dropping(self):
        """
        Returneaza True daca tendinta RSSI scade real (nu doar fluctuatie):
        media ultimei jumatati < media primei jumatati - RSSI_DROP_DELTA.
        """
        if len(self._rssi_history) < RSSI_HISTORY_SIZE:
            return False
        half = len(self._rssi_history) // 2
        old_avg = sum(self._rssi_history[:half]) / half
        new_avg = sum(self._rssi_history[half:]) / (len(self._rssi_history) - half)
        return new_avg < old_avg - RSSI_DROP_DELTA

    # ------------------ MQTT ------------------
    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            print("[Homing] MQTT conectat la broker.")
        else:
            print(f"[Homing] MQTT esuat: reason_code={reason_code}")

    def _setup_mqtt(self):
        if not MQTT_AVAILABLE:
            return False
        try:
            self.mqtt_client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2, CLIENT_ID)
            self.mqtt_client.on_connect = self._on_connect
            self.mqtt_client.connect(BROKER, PORT, 60)
            self.mqtt_client.loop_start()
            time.sleep(1.0)   # asteapta callback-ul on_connect
            return True
        except Exception as e:
            print(f"[Homing] MQTT eroare conectare: {e}")
            self.mqtt_client = None
            return False

    def _publish(self, msg):
        if self.mqtt_client is None:
            print(f"[Homing] MQTT indisponibil, skip pub: {msg}")
            return
        try:
            self.mqtt_client.publish(TOPIC, msg)
            print(f"[Homing] MQTT publicat: '{msg}'")
        except Exception as e:
            print(f"[Homing] MQTT pub eroare: {e}")

    def _cleanup_mqtt(self):
        if self.mqtt_client is not None:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except Exception:
                pass
            self.mqtt_client = None

    # ------------------ WiFi RSSI ------------------
    def _scan_rssi(self):
        """Returneaza RSSI in dBm pentru Far_Statie_Baza, sau None daca nu e vazut."""
        try:
            out = subprocess.check_output(
                ["sudo", "iwlist", "wlan0", "scan"],
                text=True, timeout=4.0,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            print("[Homing] WiFi scan timeout")
            return None
        except Exception as e:
            print(f"[Homing] WiFi scan eroare: {e}")
            return None

        # Caut "Cell ... SSID:'Far_Statie_Baza' ... Signal level=-XX"
        for cell in out.split("Cell"):
            if FAR_WIFI_SSID in cell:
                m = re.search(r"Signal level=(-?\d+)", cell)
                if m:
                    return int(m.group(1))
        return None

    # ------------------ Detectie blob LED statie ------------------
    def _detect_led_blob(self, debug=False):
        """Detecteaza LED-urile verzi ale statiei.

        LED-urile sunt foarte luminoase si satureaza camera -- apar ca pete
        galben-verzi pal. Detectia foloseste mask HSV pe verde+galben (V>=180).

        Strategie SIMPLIFICATA:
          - Cel mai mare blob > 30px = statia (LED-urile contopite la distanta)
          - bbox_w = latimea geometrica reala a conturului
          - Z = STATION_WIDTH_M (0.40 empiric) * fx / bbox_w

        Returneaza (cx_px, area_px, bbox_w_px) sau (None, 0, 0).
        Cu debug=True printeaza statistici despre contururile gasite.
        """
        try:
            frame_rgb = self.picam2.capture_array("main")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            # Corectie NoIR -- aceeasi ca in robot.py pentru consistenta detectie
            f = frame_bgr.astype(np.float32)
            f[:, :, 2] *= 1.4    # R
            f[:, :, 1] *= 1.05   # G
            f[:, :, 0] *= 0.65   # B
            np.clip(f, 0, 255, out=f)
            frame_bgr = f.astype(np.uint8)
            h, w = frame_bgr.shape[:2]

            sky_cutoff   = int(h * 0.40)
            floor_cutoff = int(h * 0.75)   # taie podeaua jos (reflexii LED)
            roi = frame_bgr[sky_cutoff:floor_cutoff, :]
            roi_area = roi.shape[0] * roi.shape[1]

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # Detectie VERDE -- un singur interval (fara wrap-around)
            mask = cv2.inRange(
                hsv,
                np.array(LED_HSV_LOW,  dtype=np.uint8),
                np.array(LED_HSV_HIGH, dtype=np.uint8),
            )
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # CLOSE dezactivat -- unea zone neconectate intr-un blob mare
            # kernel_close = np.ones((9, 9), np.uint8)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

            # Salvare debug images + frame-uri pentru dashboard live
            cv2.imwrite("/home/pi/debug_image.jpg", roi)
            cv2.imwrite("/home/pi/debug_masca.jpg", mask)
            global _last_frame_bgr, _last_mask
            with _frame_lock:
                _last_frame_bgr = roi.copy()
                _last_mask      = mask.copy()

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if debug:
                total_white = int(cv2.countNonZero(mask))
                print(f"[Statie debug] ROI={roi.shape[1]}x{roi.shape[0]} "
                      f"pixeli_albi={total_white} contururi={len(contours)}")

            if not contours:
                return None, 0, 0

            # Filtru aria minima -- accept blob-uri mari (cluster LED-uri)
            valid = []
            for c in contours:
                a = cv2.contourArea(c)
                if a < LED_MIN_AREA_DETECT:
                    if debug:
                        print(f"  blob area={a:.0f} respins (sub minim "
                              f"{LED_MIN_AREA_DETECT})")
                    continue
                valid.append((c, a))

            if not valid:
                return None, 0, 0

            # SIMPLIFICAT: cel mai mare blob verde = statia
            # (LED-urile contopite intr-o singura zona luminoasa)
            valid.sort(key=lambda x: x[1], reverse=True)
            biggest_contour, biggest_area = valid[0]

            # Prag minim pentru a fi considerat statie (nu zgomot)
            if biggest_area < 30:
                if debug:
                    print(f"  cel mai mare blob ({biggest_area:.0f}px) "
                          f"prea mic pentru statie")
                return None, 0, 0

            M = cv2.moments(biggest_contour)
            if M["m00"] == 0:
                return None, 0, 0
            bcx = M["m10"] / M["m00"]
            # Folosim bbox-ul real (latimea geometrica) -- nu estimat din aria
            # (saturarea LED-urilor face aria foarte mare fata de bbox)
            x, y, bw, bh = cv2.boundingRect(biggest_contour)
            bbox_w = bw

            if debug:
                print(f"  STATIE detectata: blob={biggest_area:.0f}px "
                      f"cx={int(bcx)} bbox_w_estim={bbox_w}px")
            global _last_info
            with _frame_lock:
                _last_info = f"STATIE: blob={int(biggest_area)}px cx={int(bcx)}"
            return int(bcx), int(biggest_area), bbox_w
        except Exception as e:
            print(f"[Homing] _detect_led_blob eroare: {e}")
            return None, 0, 0

    # ------------------ Manevre ------------------
    def _rotate_to_search(self):
        """AP pierdut sau RSSI scade -- roteste la stanga ~90deg pentru cautare."""
        print("[Homing] Cautare AP -- rotire stanga")
        self.hw["set_steering_angle"](self.hw["STEERING_LEFT_ANGLE"])
        time.sleep(0.2)
        self.hw["set_motor_pwm"]("forward", HOMING_DUTY_TURN)
        time.sleep(ROTATE_SEARCH_S)
        self.hw["set_motor_pwm"]("stop", 0.0)
        self.hw["set_steering_angle"](self.hw["STEERING_CENTER_ANGLE"])
        time.sleep(0.3)

    def _avoid_obstacle(self):
        """
        Sonar blocat -- dau cu spatele virat (alternativ stanga/dreapta),
        apoi merg inainte virat in directia opusa pentru a ocoli obstacolul.
        Acelasi pattern ca in robot.py / drive_to_waypoint.
        """
        # Alterneaza directia ocolirii fata de ultima blocare
        self._stuck_dir *= -1
        dir_name = "STANGA" if self._stuck_dir > 0 else "DREAPTA"
        print(f"[Homing] Obstacol sonar -- dau cu spatele virat {dir_name} "
              f"pana 1.5m sau sonar liber")

        # 1. Stop si vireaza pentru marsarier
        self.hw["set_motor_pwm"]("stop", 0.0)
        time.sleep(0.15)
        if self._stuck_dir > 0:
            self.hw["set_steering_angle"](self.hw["STEERING_LEFT_ANGLE"] + 5)
        else:
            self.hw["set_steering_angle"](self.hw["STEERING_RIGHT_ANGLE"] - 5)
        time.sleep(0.2)

        # 2. Marsarier pana 1.5m parcursi. NU folosim sonarul: e frontal,
        #    iar robotul merge in spate (sonar > 1.5m in fata ar opri instant).
        self.hw["set_motor_pwm"]("backward", AVOID_BACK_DUTY)
        t0_back = time.monotonic()
        while True:
            dist_back = (time.monotonic() - t0_back) * AVOID_REAL_SPEED_MS * 0.70
            if dist_back >= AVOID_BACK_MAX_DIST_M:
                print(f"[Homing] Marsarier: {dist_back:.2f}m parcursi -- stop")
                break
            # Timeout de siguranta (1.5m la ~0.07 m/s ~ 21s, punem 25)
            if time.monotonic() - t0_back > 25.0:
                print("[Homing] Marsarier: timeout siguranta -- stop")
                break
            time.sleep(0.05)
        self.hw["set_motor_pwm"]("stop", 0.0)
        time.sleep(0.2)

        # 3. Inainte virat in directia OPUSA marsarierului pentru a ocoli
        if self._stuck_dir > 0:
            # Am virat stanga la marsarier -> acum dreapta inainte
            self.hw["set_steering_angle"](self.hw["STEERING_RIGHT_ANGLE"] - 5)
            print("[Homing] Repornesc virat DREAPTA pentru ocolire")
        else:
            self.hw["set_steering_angle"](self.hw["STEERING_LEFT_ANGLE"] + 5)
            print("[Homing] Repornesc virat STANGA pentru ocolire")
        self.hw["set_motor_pwm"]("forward", AVOID_FWD_DUTY)
        # Mergi inainte ~cat ai dat cu spatele, dar limitat
        try:
            dist_back  # noqa: definita in bucla de mai sus
        except NameError:
            dist_back = AVOID_BACK_MAX_DIST_M
        fwd_t = min(max(dist_back / AVOID_REAL_SPEED_MS, AVOID_FWD_MIN_S),
                    AVOID_FWD_MAX_S)
        t0_fwd = time.monotonic()
        while time.monotonic() - t0_fwd < fwd_t:
            # Daca sonar blocat din nou, opreste si lasa bucla principala sa reia
            if self.hw["sonar_blocked"]():
                print("[Homing] Sonar blocat din nou in mers -- abandonez ocolirea")
                break
            time.sleep(0.05)

        # 4. Centrare si stop curat (bucla principala reia decizia)
        self.hw["set_steering_angle"](self.hw["STEERING_CENTER_ANGLE"])
        self.hw["set_motor_pwm"]("stop", 0.0)
        time.sleep(0.3)
        print(f"[Homing] Ocolire {dir_name} terminata -- reiau navigarea")

    def _unblock_maneuver(self):
        """
        check_stuck() a confirmat blocarea -- aceeasi manevra ca in main:
        marsarier virat alternativ pana sonar liber sau 1.5m,
        apoi inainte virat in directia OPUSA pentru a iesi din zona.
        """
        # Alterneaza directia fata de ultima blocare
        self._stuck_dir *= -1
        dir_name = "STANGA" if self._stuck_dir > 0 else "DREAPTA"
        print(f"[Homing] BLOCAJ CONFIRMAT -- dau cu spatele virat {dir_name}")

        # 1. Stop si viraj pentru marsarier
        self.hw["set_motor_pwm"]("stop", 0.0)
        time.sleep(0.3)
        if self._stuck_dir > 0:
            self.hw["set_steering_angle"](self.hw["STEERING_LEFT_ANGLE"] + 5)
        else:
            self.hw["set_steering_angle"](self.hw["STEERING_RIGHT_ANGLE"] - 5)
        time.sleep(0.2)

        # 2. Marsarier pana 1.5m parcursi (sonarul e frontal, irelevant la
        #    mers inapoi).
        self.hw["set_motor_pwm"]("backward", AVOID_BACK_DUTY)
        print("[Homing] Mers inapoi pana 1.5m parcursi...")
        t0_back   = time.monotonic()
        dist_back = 0.0
        while True:
            dist_back = (time.monotonic() - t0_back) * AVOID_REAL_SPEED_MS * 0.70
            if dist_back >= AVOID_BACK_MAX_DIST_M:
                print(f"[Homing] Mers inapoi {dist_back:.2f}m -- oprire")
                break
            # Timeout de siguranta (1.5m la ~0.07 m/s ~ 21s, punem 25)
            if time.monotonic() - t0_back > 25.0:
                print("[Homing] Marsarier: timeout siguranta -- oprire")
                break
            time.sleep(0.05)
        self.hw["set_motor_pwm"]("stop", 0.0)
        time.sleep(0.2)

        # 3. Repornire virat in directia OPUSA pentru iesire din zona
        print("[Homing] Deblocat -- repornesc virat in directie opusa")
        if self._stuck_dir > 0:
            # Marsarierul a virat stanga -> inainte vireaza dreapta
            self.hw["set_steering_angle"](self.hw["STEERING_RIGHT_ANGLE"] - 5)
        else:
            self.hw["set_steering_angle"](self.hw["STEERING_LEFT_ANGLE"] + 5)
        self.hw["set_motor_pwm"]("forward", AVOID_FWD_DUTY)
        # Timp scalat cu cat am dat cu spatele, dar minim 2s ca sa producem
        # o deplasare laterala reala (raza viraj 42cm -> 2s la 75% duty ≈
        # ~15cm lateral)
        time.sleep(min(max(dist_back / AVOID_REAL_SPEED_MS, AVOID_FWD_MIN_S),
                       AVOID_FWD_MAX_S))

        # 4. Centrare si stop -- bucla principala reia navigarea
        self.hw["set_steering_angle"](self.hw["STEERING_CENTER_ANGLE"])
        self.hw["set_motor_pwm"]("stop", 0.0)
        time.sleep(0.3)
        print(f"[Homing] Manevra de deblocare {dir_name} terminata -- "
              f"reiau navigarea")

    # ------------------ Bucla principala ------------------
    def home(self):
        """
        Naviga catre statie prin state machine FAR -> NEAR -> VISUAL_LOCK.
        Returneaza True daca a ajuns confirmat, False la timeout/eroare.
        """
        print("\n" + "=" * 60)
        print("  [HOMING] START -- intoarcere la statia de baza")
        print("  Etape: FAR (RSSI) -> NEAR (RSSI + scan LED verzi) -> drive blind")
        print("=" * 60)

        # 1. Anunta statia
        self._setup_mqtt()
        self._publish(MSG_START)
        print(f"[Homing] Astept {ESP32_BOOT_S}s -- ESP32 aprinde LED-uri + AP")
        time.sleep(ESP32_BOOT_S)

        # 2. Init state machine
        self._state                  = STATE_FAR
        self._consecutive_led_detect  = 0
        self._consecutive_led_lost    = 0
        self._consecutive_rssi_high  = 0
        self._rssi_history           = []
        t_start                      = time.monotonic()
        last_wifi_scan               = 0.0
        last_rssi                    = None

        try:
            while True:
                now = time.monotonic()

                # ====== TIMEOUT GLOBAL ======
                if now - t_start > TIMEOUT_S:
                    print(f"[Homing] TIMEOUT dupa {TIMEOUT_S:.0f}s -- abandonez")
                    self.hw["set_motor_pwm"]("stop", 0.0)
                    return False

                # ====== SIGURANTA: SONAR ======
                # In VISUAL_LOCK NU oprim pentru sonar -- il folosim ca semnal de ajungere
                if self._state != STATE_VISUAL_LOCK and self.hw["sonar_blocked"]():
                    self._avoid_obstacle()
                    last_wifi_scan = 0.0
                    continue

                # ====== SIGURANTA: STUCK DETECTION ======
                if self.hw["check_stuck"](self._stuck_state):
                    self._unblock_maneuver()
                    last_wifi_scan = 0.0
                    continue

                # ============================================================
                # STATE: VISUAL_LOCK -- LED pur (legacy), fara RSSI, rapid (10 Hz)
                # ============================================================
                if self._state == STATE_VISUAL_LOCK:
                    led_cx, led_area, led_bbox_w = self._detect_led_blob()

                    # Estimare distanta Z si lateral X din bbox-ul LED-urilor
                    if led_cx is not None and led_bbox_w > 0:
                        Z_station = (STATION_WIDTH_M * STATION_FOCAL_LENGTH) / led_bbox_w
                        X_station = (led_cx - STATION_CX0) * Z_station / STATION_FOCAL_LENGTH
                    else:
                        Z_station = None
                        X_station = None

                    # ----- Conditie de oprire: distanta Z mica -----
                    if Z_station is not None and Z_station <= STATION_STOP_DIST_M:
                        self.hw["set_motor_pwm"]("stop", 0.0)
                        self.hw["set_steering_angle"](
                            self.hw["STEERING_CENTER_ANGLE"])
                        print(f"[Homing][VL] ✓ AJUNS -- "
                              f"Z={Z_station:.2f}m <= {STATION_STOP_DIST_M}m "
                              f"(bbox_w={led_bbox_w}px X={X_station:+.2f}m)")
                        return self._confirm_arrival()

                    # ----- LED pierdut temporar? -----
                    if led_cx is None:
                        self._consecutive_led_lost += 1
                        if self._consecutive_led_lost % 3 == 0:   # log rar
                            print(f"[Homing][VL] LED pierdut "
                                  f"({self._consecutive_led_lost}/{LED_LOST_THRESHOLD})")
                        if self._consecutive_led_lost >= LED_LOST_THRESHOLD:
                            print("[Homing][VL] Lock pierdut -- revin la NEAR")
                            self._state                  = STATE_NEAR
                            self._consecutive_led_detect  = 0
                            self._consecutive_led_lost    = 0
                            self.hw["set_motor_pwm"]("stop", 0.0)
                            self._rotate_to_search()
                            last_wifi_scan = 0.0
                            continue
                        # Inca incercam -- continua drept inainte cu viteza redusa
                        self.hw["set_steering_angle"](
                            self.hw["STEERING_CENTER_ANGLE"])
                        self.hw["set_motor_pwm"]("forward", HOMING_DUTY_VISUAL)
                        time.sleep(0.1)
                        continue

                    # ----- LED detectat -- urmarire centroid (P-controller) -----
                    self._consecutive_led_lost = 0
                    err = (led_cx - IMG_CENTER_X) / float(IMG_CENTER_X)   # [-1, +1]
                    steer_offset = _clamp(err * LED_STEER_GAIN,
                                          -LED_STEER_MAX_OFF, LED_STEER_MAX_OFF)
                    steer_angle  = self.hw["STEERING_CENTER_ANGLE"] + steer_offset
                    self.hw["set_steering_angle"](steer_angle)
                    self.hw["set_motor_pwm"]("forward", HOMING_DUTY_VISUAL)
                    # Log doar la fiecare ~1s (10 iteratii * 100ms)
                    self._vl_log_counter = getattr(self, "_vl_log_counter", 0) + 1
                    if self._vl_log_counter % 10 == 0:
                        z_str = (f"Z={Z_station:.2f}m X={X_station:+.2f}m"
                                 if Z_station is not None else "Z=?")
                        print(f"[Homing][VL] cx={led_cx}px area={led_area}px "
                              f"bbox_w={led_bbox_w}px {z_str} "
                              f"err={err:+.2f} steer={steer_offset:+.1f}deg")
                    time.sleep(0.1)
                    continue

                # ============================================================
                # STATE: NEAR -- RSSI + scan LED-uri verzi continuu
                # ============================================================
                if self._state == STATE_NEAR:
                    # Scan blob LED-uri verzi la fiecare ciclu
                    led_cx, led_area, _ = self._detect_led_blob(debug=True)
                    if led_cx is not None:
                        # Opreste si confirma static (5 scan-uri rapide)
                        print(f"[Homing][NEAR] LED detectat la cx={led_cx}px area={led_area}px "
                              f"-- opresc si confirm static...")
                        self.hw["set_motor_pwm"]("stop", 0.0)
                        self.hw["set_steering_angle"](self.hw["STEERING_CENTER_ANGLE"])
                        time.sleep(0.5)   # stabilizare imagine

                        confirm_count = 0
                        for i in range(5):
                            led_cx_c, led_area_c, _ = self._detect_led_blob()
                            if led_cx_c is not None:
                                confirm_count += 1
                                print(f"  [confirm {i+1}/5] cx={led_cx_c}px area={led_area_c}px OK")
                            else:
                                print(f"  [confirm {i+1}/5] nicio detectie")
                            time.sleep(0.1)

                        if confirm_count >= 2:
                            # Calculez coordonatele DIN ULTIMUL frame confirmat
                            # si merg orbeste spre statie (fara VL care recalculeaza)
                            led_cx_f, led_area_f, led_bbox_w_f = self._detect_led_blob()
                            if led_cx_f is None or led_bbox_w_f <= 0:
                                print(f"[Homing] Confirmat dar nu am bbox final -- reiau")
                                continue
                            Z_stat = (STATION_WIDTH_M * STATION_FOCAL_LENGTH) / led_bbox_w_f
                            X_stat = (led_cx_f - STATION_CX0) * Z_stat / STATION_FOCAL_LENGTH
                            err_deg = math.degrees(math.atan2(X_stat, Z_stat))
                            print(f"\n[Homing] >>> CONFIRMAT {confirm_count}/5 -- "
                                  f"Z={Z_stat:.2f}m X={X_stat:+.2f}m err={err_deg:+.1f}deg <<<\n")

                            # Daca eroare unghi e mare, vireaza intai
                            if abs(err_deg) > 5.0:
                                print(f"[Homing] Viraj giroscop {abs(err_deg):.1f}deg "
                                      f"spre statie")
                                if err_deg > 0:
                                    self.hw["set_steering_angle"](
                                        self.hw.get("STEERING_SAFE_RIGHT", 83.0))
                                else:
                                    self.hw["set_steering_angle"](
                                        self.hw.get("STEERING_SAFE_LEFT", 25.0))
                                time.sleep(0.3)
                                self.hw["set_motor_pwm"]("forward", 80.0)
                                # Viraj pe timp (simplu)
                                dps = 6.0   # aproximare
                                turn_time = abs(err_deg) / dps
                                time.sleep(turn_time)
                                self.hw["set_motor_pwm"]("stop", 0.0)
                                self.hw["set_steering_angle"](
                                    self.hw["STEERING_CENTER_ANGLE"])
                                time.sleep(0.3)

                            # Mers drept spre statie (oprire la STATION_STOP_DIST_M)
                            dist_drive = max(0.0, Z_stat - STATION_STOP_DIST_M)
                            drive_time = dist_drive / 0.07   # 7cm/s la 60% duty
                            print(f"[Homing] Mers drept {dist_drive:.2f}m "
                                  f"({drive_time:.1f}s) catre statie")
                            self.hw["set_motor_pwm"]("forward", 60.0)
                            time.sleep(drive_time)
                            self.hw["set_motor_pwm"]("stop", 0.0)
                            print("[Homing] Ajuns la statie (drive blind)")
                            return self._confirm_arrival()
                        else:
                            print(f"[Homing][NEAR] Doar {confirm_count}/5 detectii -- "
                                  f"reiau navigarea")
                            self._consecutive_led_detect = 0
                    else:
                        self._consecutive_led_detect = 0

                    # Scan RSSI la WIFI_SCAN_INTERVAL_S (blocheaza ~1-3s)
                    if now - last_wifi_scan >= WIFI_SCAN_INTERVAL_S:
                        last_wifi_scan = now
                        rssi = self._scan_rssi()

                        if rssi is None:
                            if now - t_start < 5.0:
                                print("[Homing][NEAR] AP nedetectat (grace period) "
                                      "-- merg drept")
                                self.hw["set_steering_angle"](
                                    self.hw["STEERING_CENTER_ANGLE"])
                                self.hw["set_motor_pwm"]("forward", HOMING_DUTY_NEAR_HIGH)
                                time.sleep(0.1)
                                continue
                            print("[Homing][NEAR] Far_Statie_Baza pierdut")
                            self.hw["set_motor_pwm"]("stop", 0.0)
                            self._rotate_to_search()
                            last_rssi = None
                            self._rssi_history = []
                            continue

                        self._push_rssi(rssi)
                        trend = ("↘" if self._rssi_is_dropping()
                                 else "↗" if (last_rssi is not None
                                              and rssi > last_rssi + 2) else "→")
                        print(f"[Homing][NEAR] RSSI={rssi} dBm {trend} "
                              f"hist={self._rssi_history}")

                        # FALLBACK: RSSI foarte bun dar LED-uri invizibile
                        # (probabil HSV slab calibrat) -> fortez VISUAL_LOCK
                        if rssi >= RSSI_FORCE_VL_THRESHOLD:
                            self._consecutive_rssi_high += 1
                            print(f"[Homing][NEAR] RSSI foarte bun "
                                  f"({self._consecutive_rssi_high}/"
                                  f"{RSSI_FORCE_VL_COUNT}) -- "
                                  f"posibil fortez VISUAL_LOCK")
                            if self._consecutive_rssi_high >= RSSI_FORCE_VL_COUNT:
                                # Salveaza un frame debug ca sa vedem
                                # de ce LED-urile nu se detecteaza la RSSI atat de bun
                                try:
                                    fr = self.picam2.capture_array("main")
                                    fr_bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                                    cv2.imwrite("/home/pi/led_debug.png", fr_bgr)
                                    print("[Homing] Frame debug salvat: "
                                          "/home/pi/led_debug.png "
                                          "(verifica de ce nu vede LED-uri la RSSI bun)")
                                except Exception as e:
                                    print(f"[Homing] Nu am putut salva debug: {e}")
                                print(f"\n[Homing] >>> TRANZITIE FORTATA "
                                      f"NEAR -> VISUAL_LOCK (RSSI={rssi}) <<<\n")
                                self._state                = STATE_VISUAL_LOCK
                                self._consecutive_led_lost  = 0
                                self.hw["set_steering_angle"](
                                    self.hw["STEERING_CENTER_ANGLE"])
                                last_rssi = rssi
                                continue
                        else:
                            self._consecutive_rssi_high = 0

                        # Reorientare doar daca tendinta clara scade
                        if self._rssi_is_dropping() and (now - t_start) >= 5.0:
                            print(f"[Homing][NEAR] >>> Tendinta scade -- "
                                  f"reorientare 90deg stanga <<<")
                            self.hw["set_motor_pwm"]("stop", 0.0)
                            self._rotate_to_search()
                            self._rssi_history = []   # reset dupa rotire
                            last_rssi = None
                            continue
                        else:
                            self.hw["set_steering_angle"](
                                self.hw["STEERING_CENTER_ANGLE"])

                        # Scalare progresiva viteza: cu cat RSSI mai bun,
                        # cu atat mai incet (sa avem timp sa detectam LED-urile)
                        if rssi <= NEAR_RSSI_FAST_DBM:
                            duty = HOMING_DUTY_NEAR_HIGH
                        elif rssi >= NEAR_RSSI_SLOW_DBM:
                            duty = HOMING_DUTY_NEAR_LOW
                        else:
                            # Interpolare liniara intre HIGH si LOW
                            frac = ((rssi - NEAR_RSSI_FAST_DBM) /
                                    (NEAR_RSSI_SLOW_DBM - NEAR_RSSI_FAST_DBM))
                            duty = (HOMING_DUTY_NEAR_HIGH +
                                    frac * (HOMING_DUTY_NEAR_LOW -
                                            HOMING_DUTY_NEAR_HIGH))
                        print(f"[Homing][NEAR] viteza={duty:.0f}% "
                              f"(RSSI={rssi} -> scalare)")
                        self.hw["set_motor_pwm"]("forward", duty)
                        last_rssi = rssi

                    time.sleep(0.05)
                    continue

                # ============================================================
                # STATE: FAR -- doar RSSI, viteza mare
                # ============================================================
                if self._state == STATE_FAR:
                    if now - last_wifi_scan >= WIFI_SCAN_INTERVAL_S:
                        last_wifi_scan = now
                        rssi = self._scan_rssi()

                        if rssi is None:
                            # Grace period: in primele 5 secunde, doar merge drept
                            # fara sa caute (evita rotire stanga la start)
                            if now - t_start < 5.0:
                                print(f"[Homing][FAR] AP nedetectat (grace period) -- "
                                      f"merg drept")
                                self.hw["set_steering_angle"](
                                    self.hw["STEERING_CENTER_ANGLE"])
                                self.hw["set_motor_pwm"]("forward", HOMING_DUTY_FAR)
                                time.sleep(0.1)
                                continue
                            print("[Homing][FAR] Far_Statie_Baza pierdut")
                            self.hw["set_motor_pwm"]("stop", 0.0)
                            self._rotate_to_search()
                            last_rssi = None
                            self._rssi_history = []
                            continue

                        # Tranzitie FAR -> NEAR
                        if rssi >= RSSI_NEAR_THRESHOLD:
                            print(f"\n[Homing] >>> TRANZITIE FAR -> NEAR "
                                  f"(RSSI={rssi} dBm) <<<\n")
                            self._state                  = STATE_NEAR
                            self._consecutive_led_detect  = 0
                            self._rssi_history           = []
                            last_rssi                    = rssi
                            continue

                        self._push_rssi(rssi)
                        trend = ("↘" if self._rssi_is_dropping()
                                 else "↗" if (last_rssi is not None
                                              and rssi > last_rssi + 2) else "→")
                        print(f"[Homing][FAR] RSSI={rssi} dBm {trend} "
                              f"hist={self._rssi_history}")

                        # Reorientare doar daca tendinta clara scade
                        if self._rssi_is_dropping():
                            print(f"[Homing][FAR] >>> Tendinta scade -- "
                                  f"reorientare 90deg stanga <<<")
                            self.hw["set_motor_pwm"]("stop", 0.0)
                            self._rotate_to_search()
                            self._rssi_history = []
                            last_rssi = None
                            continue
                        else:
                            self.hw["set_steering_angle"](
                                self.hw["STEERING_CENTER_ANGLE"])

                        self.hw["set_motor_pwm"]("forward", HOMING_DUTY_FAR)
                        last_rssi = rssi

                    time.sleep(0.1)
                    continue

        except KeyboardInterrupt:
            print("[Homing] Intrerupt utilizator")
            return False
        finally:
            # Asigura-te ca motorul e oprit la iesire
            try:
                self.hw["set_motor_pwm"]("stop", 0.0)
                self.hw["set_steering_angle"](self.hw["STEERING_CENTER_ANGLE"])
            except Exception:
                pass

    def _confirm_arrival(self):
        """Verificare finala dupa stop in VISUAL_LOCK -- publica AJUNS pe MQTT."""
        self.hw["set_motor_pwm"]("stop", 0.0)
        self.hw["set_steering_angle"](self.hw["STEERING_CENTER_ANGLE"])
        time.sleep(0.5)
        # O ultima verificare a blob-ului LED pentru log final
        final_cx, final_area, _ = self._detect_led_blob()
        print(f"[Homing] ✓✓ ARRIVAL CONFIRMED -- "
              f"final LED: cx={final_cx} area={final_area}px")
        if rtc_module is not None:
            rtc_module.log_event("HOMING_ARRIVED",
                f"Arrival confirmed (cx={final_cx} area={final_area}px)")
        self._publish(MSG_ARRIVED)
        return True

    def signal_unloaded(self):
        """Anunta statia ca robotul e descarcat -- stinge LED-uri + AP."""
        self._publish(MSG_UNLOADED)
        time.sleep(0.5)


# ========================================================
# API pentru integrare in main
# ========================================================
def home_to_station(picam2, hw, wait_after_arrive=True):
    """
    Functie publica apelata din robot.py dupa ce colectorul e plin.

    Parametri:
      picam2  -- instanta Picamera2 deja initializata
      hw      -- dict cu handles hardware (vezi HomingController.__init__)
      wait_after_arrive -- daca True, asteapta ARRIVE_WAIT_S la baza si trimite GOL

    Returneaza True daca a ajuns la statie (succes), False altfel.
    """
    # Pornesc dashboard-ul live pe port 5002 (idempotent -- doar prima oara)
    if not hasattr(home_to_station, "_dashboard_started"):
        start_homing_dashboard(port=5002)
        home_to_station._dashboard_started = True
    ctrl = HomingController(picam2, hw)
    success = False
    try:
        success = ctrl.home()
        if success and wait_after_arrive:
            print(f"\n[Homing] Astept {ARRIVE_WAIT_S:.0f}s pentru "
                  f"descarcare manuala...")
            # Asigura motorul oprit pe toata durata asteptarii
            hw["set_motor_pwm"]("stop", 0.0)
            hw["set_steering_angle"](hw["STEERING_CENTER_ANGLE"])
            time.sleep(ARRIVE_WAIT_S)
            ctrl.signal_unloaded()
            print("[Homing] Reluare patrulare\n")
    finally:
        ctrl._cleanup_mqtt()
    return success


# ========================================================
# FLASK DASHBOARD LIVE (port 5002)
# ========================================================
def start_homing_dashboard(port=5002):
    """Porneste un mic server Flask cu stream live al detectiei statiei."""
    if not FLASK_AVAILABLE:
        print("[Homing Dashboard] Flask nu e instalat -- skip")
        return

    app = Flask(__name__)

    HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Homing Live</title>
<style>
body { font-family:sans-serif; background:#f5f7fa; color:#2d3748; padding:1.5rem; margin:0; }
h1 { font-size:1.2rem; color:#0a8055; text-align:center; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:1rem; max-width:1200px; margin:0 auto; }
.card { background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1rem; box-shadow:0 1px 3px rgba(0,0,0,0.05); }
.card h3 { margin-top:0; color:#4a5568; font-size:1rem; }
.info { background:#0a8055; color:#fff; padding:0.5rem 1rem; border-radius:6px; font-family:monospace; text-align:center; max-width:1200px; margin:1rem auto; }
img { width:100%; border-radius:4px; }
</style></head><body>
<h1>Homing Live -- Detectie Statie</h1>
<div class="info" id="info">Astept...</div>
<div class="grid">
<div class="card"><h3>Imagine ROI (cu corectie NoIR)</h3><img src="/stream_image" /></div>
<div class="card"><h3>Masca HSV (alb = verde detectat)</h3><img src="/stream_mask" /></div>
</div>
<script>
setInterval(async () => {
    const r = await fetch('/info');
    const t = await r.text();
    document.getElementById('info').textContent = t;
}, 500);
</script>
</body></html>"""

    def gen_image():
        while True:
            with _frame_lock:
                f = _last_frame_bgr.copy() if _last_frame_bgr is not None else None
            if f is None:
                time.sleep(0.1); continue
            ret, jpeg = cv2.imencode(".jpg", f)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(0.1)

    def gen_mask():
        while True:
            with _frame_lock:
                m = _last_mask.copy() if _last_mask is not None else None
            if m is None:
                time.sleep(0.1); continue
            ret, jpeg = cv2.imencode(".jpg", m)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(0.1)

    @app.route("/")
    def index():
        return render_template_string(HTML)

    @app.route("/stream_image")
    def stream_image():
        return Response(gen_image(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/stream_mask")
    def stream_mask():
        return Response(gen_mask(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/info")
    def info():
        with _frame_lock:
            return _last_info

    t = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=port,
                                                 threaded=True, debug=False),
                         daemon=True)
    t.start()
    print(f"[Homing Dashboard] Ruleaza la http://0.0.0.0:{port}")


# ========================================================
# MODE STANDALONE (testare directa: python3 robot_homing.py)
# ========================================================
if __name__ == "__main__":
    """
    Testare directa fara robot.py. Initializeaza hardware-ul minim
    si ruleaza un homing complet.
    """
    import RPi.GPIO as GPIO
    from adafruit_servokit import ServoKit
    import board, busio
    from picamera2 import Picamera2

    # Configuratie minim necesara (copiata din robot.py pentru standalone)
    PIN_RPWM = 33
    PIN_LPWM = 32
    PWM_FREQ = 1000
    STEERING_CHANNEL      = 0
    STEERING_CENTER_ANGLE = 60.0
    STEERING_LEFT_ANGLE   = 30.0
    STEERING_RIGHT_ANGLE  = 95.0
    SONAR_TRIG_PIN = 18
    SONAR_ECHO_PIN = 16
    SONAR_STOP_M   = 0.50

    # Init GPIO
    try:
        GPIO.cleanup()
    except Exception:
        pass
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(PIN_RPWM, GPIO.OUT)
    GPIO.setup(PIN_LPWM, GPIO.OUT)
    GPIO.setup(SONAR_TRIG_PIN, GPIO.OUT)
    GPIO.setup(SONAR_ECHO_PIN, GPIO.IN)
    GPIO.output(SONAR_TRIG_PIN, False)

    pwm_r = GPIO.PWM(PIN_RPWM, PWM_FREQ)
    pwm_l = GPIO.PWM(PIN_LPWM, PWM_FREQ)
    pwm_r.start(0)
    pwm_l.start(0)

    i2c = busio.I2C(board.SCL, board.SDA)
    kit = ServoKit(channels=16, i2c=i2c)
    kit.servo[STEERING_CHANNEL].angle = STEERING_CENTER_ANGLE

    # Init camera
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(cfg)
    picam2.start()
    time.sleep(1.0)

    # Functii hardware locale
    def _set_motor_pwm(direction, duty):
        duty = max(0.0, min(100.0, float(duty)))
        if direction == "forward":
            pwm_l.ChangeDutyCycle(0.0)
            pwm_r.ChangeDutyCycle(duty)
        elif direction == "backward":
            pwm_r.ChangeDutyCycle(0.0)
            pwm_l.ChangeDutyCycle(duty)
        else:
            pwm_r.ChangeDutyCycle(0.0)
            pwm_l.ChangeDutyCycle(0.0)

    def _set_steering(angle):
        angle = max(0.0, min(180.0, float(angle)))
        try:
            kit.servo[STEERING_CHANNEL].angle = angle
        except Exception as e:
            print(f"[Standalone] Servo eroare: {e}")

    def _stop_and_center():
        _set_motor_pwm("stop", 0.0)
        _set_steering(STEERING_CENTER_ANGLE)

    def _read_sonar():
        try:
            GPIO.output(SONAR_TRIG_PIN, True)
            time.sleep(0.00001)
            GPIO.output(SONAR_TRIG_PIN, False)
            deadline = time.monotonic() + 0.025
            while GPIO.input(SONAR_ECHO_PIN) == 0:
                if time.monotonic() > deadline: return 99.0
            t0 = time.monotonic()
            deadline = time.monotonic() + 0.025
            while GPIO.input(SONAR_ECHO_PIN) == 1:
                if time.monotonic() > deadline: return 99.0
            t1 = time.monotonic()
            return (t1 - t0) * 343.0 / 2.0
        except Exception:
            return 99.0

    _sonar_recent = []
    def _sonar_loop():
        while True:
            d = _read_sonar()
            _sonar_recent.append(d)
            if len(_sonar_recent) > 5:
                _sonar_recent.pop(0)
            time.sleep(0.1)

    threading.Thread(target=_sonar_loop, daemon=True).start()
    time.sleep(0.5)

    def _sonar_blocked():
        if len(_sonar_recent) < 3: return False
        valid = [d for d in _sonar_recent if d < 90.0]
        if len(valid) < 3: return False
        return sum(1 for d in valid if 0.0 < d < SONAR_STOP_M) >= 3

    def _sonar_m_now():
        """Distanta sonar curenta in metri (99.0 daca nu vede)."""
        if not _sonar_recent:
            return 99.0
        return _sonar_recent[-1]

    def _check_stuck(state):
        return False   # standalone -- fara MPU6050

    hw = {
        "set_motor_pwm":          _set_motor_pwm,
        "set_steering_angle":     _set_steering,
        "stop_and_center":        _stop_and_center,
        "sonar_blocked":          _sonar_blocked,
        "sonar_m_now":            _sonar_m_now,
        "check_stuck":            _check_stuck,
        "state_set":              lambda **kw: None,
        "STEERING_CENTER_ANGLE":  STEERING_CENTER_ANGLE,
        "STEERING_LEFT_ANGLE":    STEERING_LEFT_ANGLE,
        "STEERING_RIGHT_ANGLE":   STEERING_RIGHT_ANGLE,
    }

    print("=== TEST STANDALONE HOMING ===")
    try:
        ok = home_to_station(picam2, hw, wait_after_arrive=False)
        print(f"\n[Standalone] Rezultat: {'SUCCES' if ok else 'ESEC'}")
    finally:
        print("\n[Standalone] Cleanup...")
        try:
            _stop_and_center()
            time.sleep(0.2)
            pwm_r.stop()
            pwm_l.stop()
        except Exception:
            pass
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            GPIO.cleanup()
        except Exception:
            pass
        print("=== STOP ===")
