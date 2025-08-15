ONE_KNOT_IN_MS = 0.51444444 # 1kn = 0.5144444 m/s

def ms_to_knot(speed_ms:float) -> float:
    return speed_ms / ONE_KNOT_IN_MS

def knot_to_ms(speed_kn:float) -> float:
    return speed_kn * ONE_KNOT_IN_MS 