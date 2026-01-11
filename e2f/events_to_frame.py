from pathlib import Path
from scripts.utils.functions import *

if __name__ == '__main__':
    event_filepath = Path("events.h5")
    event_delta_time_ms = 100.0
    high_low_ratio = 10.0
    rgb_delta_time_ms = 50.0
    use_time_lookup = False
    use_calibration = True

    time_surface_str = f"_custom_time_surface_ratio_{int(high_low_ratio)}_power_v2" if use_time_lookup else ""
    calibration_str = "_calibrated" if use_calibration else ""
    output_path = Path(f"output_event_{event_delta_time_ms}ms{calibration_str}{time_surface_str}_{rgb_delta_time_ms}_rgb_ms")
    

    event_to_frame(event_filepath, event_delta_time_ms, high_low_ratio, rgb_delta_time_ms, use_time_lookup, use_calibration, output_path, calibration_file=Path("cam_to_cam.yaml"), gamma=2.0)