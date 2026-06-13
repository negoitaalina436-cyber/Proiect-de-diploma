"""
rtc_module.py -- Modul de gestionare RTC DS3231 pentru robotul de tenis.

Funcționalități:
  1. Citire oră curentă din DS3231 (I2C 0x68)
  2. Setare oră sistem din RTC la pornire (când nu există internet)
  3. Timestamp pentru log-uri
  4. Salvare jurnal evenimente în fișier
"""

import os
import time
import datetime
import threading

try:
    import smbus2 as smbus
except ImportError:
    try:
        import smbus
    except ImportError:
        smbus = None

DS3231_ADDR = 0x68
JOURNAL_DIR = "/home/pi/journal"
JOURNAL_FILE = None   # setat la init


def _bcd_to_int(b):
    return (b & 0x0F) + ((b >> 4) & 0x07) * 10


def _int_to_bcd(n):
    return ((n // 10) << 4) | (n % 10)


def read_rtc():
    """Citeste data si ora curenta de la DS3231.
    Returneaza datetime sau None la eroare."""
    if smbus is None:
        return None
    try:
        bus = smbus.SMBus(1)
        data = bus.read_i2c_block_data(DS3231_ADDR, 0x00, 7)
        bus.close()
        sec  = _bcd_to_int(data[0] & 0x7F)
        mn   = _bcd_to_int(data[1] & 0x7F)
        hr   = _bcd_to_int(data[2] & 0x3F)
        day  = _bcd_to_int(data[4] & 0x3F)
        mo   = _bcd_to_int(data[5] & 0x1F)
        yr   = _bcd_to_int(data[6]) + 2000
        return datetime.datetime(yr, mo, day, hr, mn, sec)
    except Exception as e:
        print(f"[RTC] Eroare citire: {e}")
        return None


def write_rtc(dt):
    """Scrie data/ora la DS3231."""
    if smbus is None:
        return False
    try:
        bus = smbus.SMBus(1)
        data = [
            _int_to_bcd(dt.second),
            _int_to_bcd(dt.minute),
            _int_to_bcd(dt.hour),
            _int_to_bcd(dt.isoweekday()),
            _int_to_bcd(dt.day),
            _int_to_bcd(dt.month),
            _int_to_bcd(dt.year % 100),
        ]
        bus.write_i2c_block_data(DS3231_ADDR, 0x00, data)
        bus.close()
        return True
    except Exception as e:
        print(f"[RTC] Eroare scriere: {e}")
        return False


def sync_system_clock_from_rtc():
    """Citeste RTC si seteaza ceasul sistemului. Util cand nu e internet."""
    dt = read_rtc()
    if dt is None:
        print("[RTC] Nu am putut citi -- skip sincronizare sistem")
        return False
    # Verifica daca diferenta e semnificativa (>5 secunde)
    sys_now = datetime.datetime.now()
    diff = abs((sys_now - dt).total_seconds())
    if diff < 5:
        print(f"[RTC] Sistem deja sincronizat (diff={diff:.1f}s) -- skip")
        return True
    # Setare ceas sistem (necesita root sau sudo)
    try:
        date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        ret = os.system(f"sudo date -s '{date_str}' > /dev/null 2>&1")
        if ret == 0:
            print(f"[RTC] Ceas sistem setat din RTC: {date_str}")
            return True
        else:
            print(f"[RTC] Esec setare ceas sistem (cod {ret})")
            return False
    except Exception as e:
        print(f"[RTC] Eroare set ceas sistem: {e}")
        return False


def now_str():
    """Returneaza timestamp formatat pentru log-uri: '[12:34:56]'."""
    dt = read_rtc()
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime("[%H:%M:%S]")


def now_full():
    """Timestamp complet pentru jurnal: '2026-06-11 12:34:56'."""
    dt = read_rtc()
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ==========================================================
# JURNAL EVENIMENTE
# ==========================================================
_journal_lock = threading.Lock()


def journal_init():
    """Creeaza fisierul jurnal nou la pornirea robotului."""
    global JOURNAL_FILE
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    dt = read_rtc() or datetime.datetime.now()
    fname = dt.strftime("robot_%Y-%m-%d_%H-%M-%S.log")
    JOURNAL_FILE = os.path.join(JOURNAL_DIR, fname)
    try:
        with open(JOURNAL_FILE, "w") as f:
            f.write(f"# Jurnal robot tenis -- start {now_full()}\n")
            f.write(f"# {'='*60}\n")
        print(f"[Jurnal] Initializat: {JOURNAL_FILE}")
    except Exception as e:
        print(f"[Jurnal] Eroare init: {e}")
        JOURNAL_FILE = None


def log_event(category, message):
    """Salveaza un eveniment in jurnal cu timestamp.
    
    Categorii sugerate:
      START, PATROL, BALL_DETECT, BALL_COLLECT, HOMING_START,
      HOMING_ARRIVED, ERROR, STUCK, etc.
    """
    if JOURNAL_FILE is None:
        return
    ts = now_full()
    line = f"{ts} [{category:>15}] {message}\n"
    with _journal_lock:
        try:
            with open(JOURNAL_FILE, "a") as f:
                f.write(line)
        except Exception as e:
            print(f"[Jurnal] Eroare scriere: {e}")


def log_session_summary(balls_collected, duration_s, success=True):
    """La final sesiune, scrie un sumar."""
    if JOURNAL_FILE is None:
        return
    summary = f"""
# {'='*60}
# SUMAR SESIUNE -- {now_full()}
# {'='*60}
# Mingi colectate:  {balls_collected}
# Durata:           {duration_s:.1f}s ({duration_s/60:.1f}min)
# Rezultat:         {'SUCCES' if success else 'ESEC'}
"""
    with _journal_lock:
        try:
            with open(JOURNAL_FILE, "a") as f:
                f.write(summary)
            print(f"[Jurnal] Sumar salvat")
        except Exception as e:
            print(f"[Jurnal] Eroare sumar: {e}")


# ==========================================================
# TESTING STANDALONE
# ==========================================================
if __name__ == "__main__":
    print("=== TEST RTC DS3231 ===")
    dt = read_rtc()
    if dt:
        print(f"Ora RTC: {dt}")
    else:
        print("Nu am putut citi RTC")
    
    print(f"now_str:  {now_str()}")
    print(f"now_full: {now_full()}")
    
    journal_init()
    log_event("TEST", "Test eveniment 1")
    log_event("BALL_COLLECT", "Minge colectata la (1.2, 0.5)m")
    log_event("HOMING_START", "Start homing dupa 5 mingi")
    log_session_summary(5, 120.5)
    print("Test complet.")
