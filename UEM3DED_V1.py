#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pathlib
import re
import requests
import json
import shutil
import statistics
import sys
import time
import threading
import traceback
import socket
import numpy as np
import matplotlib.pyplot as plt
import newportESP
import datetime

__author__ = "Amsterdam Scientific Instruments"
__version__ = "2.1"
__email__ = "support@amscins.com"

CONFIG_FILE = pathlib.Path(__file__).with_name("UEM3DED_config.txt")

DEFAULT_CONFIG = {
    "WORK_DIR": "/home/asi/Software/ASI/20211004 ASI Server 2.3.0 (TPX3)/examples/tpx3/",
    "TEM_HOST": "172.17.41.1",
    "TEM_PORT": "9090",
    "SERVAL_URL": "http://localhost:8080",
    "NEWPORT_PORT": "/dev/ttyUSB0",
    "ASI_ADDRESS": "127.0.0.1",
    "ASI_PORT1": "6351",
    "ASI_PORT2": "6352",
    "BPC_FILE": "/home/asi/Desktop/Factory settings 23-11-2023/factory-test-config/high-power/eq-accos-dd-04.bpc",
    "DACS_FILE": "/home/asi/Desktop/Factory settings 23-11-2023/factory-test-config/high-power/eq-accos-dd-05_200keV-e.dacs",
    "WAVELENGTH": "0.025079",
    "CCD_PIXEL_SIZE": "0.007249",
    "ROTATION_AXIS": "117.4",
    "BEAM_TILT_STEP": "0",
    "BEAM_TILT_RANGE": "0.0",
    "STRETCHING_MP": "0.0",
    "STRETCHING_AZIMUTH": "0.0",
}


def load_text_config(path):
    config = DEFAULT_CONFIG.copy()

    if not path.exists():
        print("[Config] Config file not found, using built-in defaults: %s" % path)
        return config

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                print("[Config] Ignoring line %d without '=': %s" % (line_number, line))
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key:
                config[key] = value

    return config


CONFIG = load_text_config(CONFIG_FILE)

WORK_DIR = CONFIG["WORK_DIR"]
TEM_HOST = CONFIG["TEM_HOST"]
TEM_PORT = int(CONFIG["TEM_PORT"])
SERVAL_URL = CONFIG["SERVAL_URL"]
NEWPORT_PORT = CONFIG["NEWPORT_PORT"]
ASI_ADDRESS = CONFIG["ASI_ADDRESS"]
ASI_PORT1 = int(CONFIG["ASI_PORT1"])
ASI_PORT2 = int(CONFIG["ASI_PORT2"])
BPC_FILE = CONFIG["BPC_FILE"]
DACS_FILE = CONFIG["DACS_FILE"]
DELAY_STAGE_SETTLE_TIME = 1.0
TEM_TILT_SETTLE_TIME = 1.0

# =========================
# TEM control from Ubuntu
# =========================


def send_tem_command(command, timeout=60):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect((TEM_HOST, TEM_PORT))
        s.sendall((command + "\n").encode("utf-8"))
        response = s.recv(4096).decode("utf-8").strip()

    if not response.startswith("OK"):
        raise RuntimeError(f"TEM command failed: {response}")

    return response


def tem_get_alpha():
    response = send_tem_command("GET_ALPHA")
    return float(response.split()[-1])


def tem_tilt(angle):
    response = send_tem_command(f"TILT {float(angle)}")
    return float(response.split()[-1])


# Newport delay stage
esp = newportESP.ESP(NEWPORT_PORT)
stage = esp.axis(1)

# ASI socket
address = ASI_ADDRESS
port1 = ASI_PORT1
rcv_buffer_size = 4096
port2 = ASI_PORT2
sock1 = socket.socket()

try:
    sock1.connect((address, port1))
except ConnectionRefusedError:
    print("\nCONNECTION REFUSED: THE SOFTWARE IS NOT RUNNING OR THE ADDRESS OR PORT ARE WRONG/BLOCKED\n")
except BrokenPipeError:
    print("\nBroken Pipe error")


from matplotlib.backends.qt_compat import QtCore, QtWidgets, QtGui, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
else:
    from matplotlib.backends.backend_qt4agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


def get_request(url, expected_status=200):
    response = requests.get(url=url)
    while response.status_code == 409:
        time.sleep(1)
        response = requests.get(url=url)

    if response.status_code != expected_status:
        raise Exception("Failed GET request: {}, response: {} {}".format(
            url, response.status_code, response.text
        ))

    return response


def put_request(url, data, expected_status=200):
    response = requests.put(url=url, data=data)

    if response.status_code != expected_status:
        raise Exception("Failed PUT request: {}, response: {} {}".format(
            url, response.status_code, response.text
        ))

    return response


def check_connection(serverurl):
    get_request(url=serverurl, expected_status=200)


def get_dashboard(serverurl):
    response = get_request(url=serverurl + '/dashboard')
    return json.loads(response.text)


def get_detectorconfig(serverurl):
    response = get_request(url=serverurl + '/detector/config')
    return json.loads(response.text)


def init_cam(serverurl, bpc_file, dacs_file):
    response = get_request(url=serverurl + '/config/load?format=pixelconfig&file=' + bpc_file)
    print('Response of loading binary pixel configuration file: ' + response.text)

    response = get_request(url=serverurl + '/config/load?format=dacs&file=' + dacs_file)
    print('Response of loading DACs file: ' + response.text)


def init_acquisition(serverurl, detector_config, ntriggers=1, trigger_period=0.5, exposure_time=0.10):
    detector_config["nTriggers"] = ntriggers
    detector_config["TriggerMode"] = "AUTOTRIGSTART_TIMERSTOP"
    detector_config["TriggerPeriod"] = trigger_period
    detector_config["ExposureTime"] = exposure_time

    response = put_request(url=serverurl + '/detector/config', data=json.dumps(detector_config))
    print('Response of updating Detector Configuration: ' + response.text)


def simple_acquisition(serverurl):
    response = get_request(url=serverurl + '/measurement/start')
    print('Response of acquisition start: ' + response.text)
    print('current dirrectory: ' + os.getcwd())


def append_acquisition_log(directory, angle, actual_alpha, delay, loop, renamed_files):
    log_path = os.path.join(directory, "acquisition_log.jsonl")
    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "angle_deg": float(angle),
        "actual_alpha_deg": float(actual_alpha),
        "delay_ps": float(delay),
        "loop": int(loop),
        "files": [os.path.basename(path) for path in renamed_files],
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def file_rename(directory, delay, loop, count, timeout=120):
    target = os.path.abspath(directory)

    num_frame = 0
    file_count = 0
    deadline = time.time() + timeout

    while file_count < count:
        file_count = 0
        allfiles = os.listdir(target)

        for filename in allfiles:
            if "loop" in filename:
                continue
            if not os.path.isfile(os.path.join(target, filename)):
                continue
            file_count += 1

        if time.time() > deadline:
            raise TimeoutError(
                "Timed out waiting for %d files in %s; found %d after %d seconds"
                % (count, target, file_count, timeout)
            )

        time.sleep(1)

    allfiles = os.listdir(target)
    renamed_files = []

    for filename in allfiles:
        old_path = os.path.join(target, filename)

        if not os.path.isfile(old_path):
            continue
        if "loop" in filename:
            continue

        t = os.path.getmtime(old_path)
        v = datetime.datetime.fromtimestamp(t)
        x = v.strftime('%Y%m%d%H%M%S')
        oldext = os.path.splitext(filename)[1]

        if oldext != '.json':
            temp_name = (
                x + "_loop " + str(loop).zfill(2) +
                "_" + str(delay) + "ps_f " +
                str(num_frame).zfill(2) + oldext
            )
            new_path = os.path.join(target, temp_name)
            os.rename(old_path, new_path)
            renamed_files.append(new_path)
            num_frame += 1

        else:
            temp_name = (
                x + "_loop " + str(loop).zfill(2) +
                "_" + str(delay) + "ps" + oldext
            )
            new_path = os.path.join(target, temp_name)
            os.rename(old_path, new_path)
            renamed_files.append(new_path)

    return renamed_files


def parse_angle_from_directory_name(name):
    if name.startswith("angle_"):
        name = name[len("angle_"):]
    return float(name)


def build_3ded_reconstruction(
    src_root,
    dst_root=None,
    wavelength=0.025079,
    ccd_pixel_size=0.007249,
    rotation_axis=117.4,
    beam_tilt_step=0,
    beam_tilt_range=0.0,
    stretching_mp=0.0,
    stretching_azimuth=0.0,
):
    import mrcfile
    import tifffile

    src_root = pathlib.Path(src_root)
    if dst_root is None:
        dst_root = src_root / (src_root.name + "_3DED")
    else:
        dst_root = pathlib.Path(dst_root)

    delay_pattern = re.compile(r"([+-]?[0-9]+(?:\.[0-9]+)?)ps")
    tiff_files = list(src_root.rglob("*.tiff"))

    delay_map = {}
    for f in tiff_files:
        m = delay_pattern.search(f.name)
        if not m:
            continue
        delay = float(m.group(1))
        delay_map.setdefault(delay, []).append(f)

    sorted_delays = sorted(delay_map.keys())
    if not sorted_delays:
        raise RuntimeError("No delay-labelled TIFF files found under %s" % src_root)

    delay_to_exp = {d: "experiment_%d" % (i + 1) for i, d in enumerate(sorted_delays)}

    for delay in sorted_delays:
        exp = delay_to_exp[delay]
        exp_dir = dst_root / exp
        tiff_out = exp_dir / "tiff"
        red_out = exp_dir / "RED"

        tiff_out.mkdir(parents=True, exist_ok=True)
        red_out.mkdir(parents=True, exist_ok=True)

        angle_files = []
        for f in delay_map[delay]:
            angle = parse_angle_from_directory_name(f.parent.name)
            angle_files.append((angle, f))
        angle_files.sort(key=lambda x: x[0])

        angles = []
        with (exp_dir / "angles.txt").open("w", encoding="utf-8") as af:
            for idx, (angle, src_tiff) in enumerate(angle_files, start=1):
                name = "%05d" % idx
                angles.append(angle)
                af.write("%.6f\n" % angle)

                shutil.copy2(src_tiff, tiff_out / (name + ".tiff"))

                img = tifffile.imread(src_tiff)
                if img.dtype == np.uint16:
                    img_u16 = img
                else:
                    img_u16 = np.clip(img, 0, 65535).astype(np.uint16)

                with mrcfile.new(red_out / (name + ".mrc"), overwrite=True) as mrc:
                    mrc.set_data(img_u16)

        diffs = [
            angles[i + 1] - angles[i]
            for i in range(len(angles) - 1)
            if abs(angles[i + 1] - angles[i]) > 1e-6
        ]
        stepsize = statistics.median(diffs) if diffs else 0.0

        with (exp_dir / "experiment_info.txt").open("w", encoding="utf-8") as f:
            f.write(
                "Experiment: %s\n"
                "Delay time: %.3f ps\n\n"
                "Auto-derived parameters:\n"
                "Start angle: %.2f degrees\n"
                "End angle: %.2f degrees\n"
                "Rotation range: %.2f degrees\n"
                "Stepsize: %.4f degrees\n"
                "Number of frames: %d\n\n"
                "(Other experimental parameters to be filled manually)\n"
                % (
                    exp,
                    delay,
                    angles[0],
                    angles[-1],
                    angles[-1] - angles[0],
                    stepsize,
                    len(angles),
                )
            )

        with (red_out / (exp + ".ed3d")).open("w", encoding="utf-8") as fout:
            fout.write("WAVELENGTH    %.6f\n" % wavelength)
            fout.write("ROTATIONAXIS    %.6f\n" % rotation_axis)
            fout.write("CCDPIXELSIZE    %.6f\n" % ccd_pixel_size)
            fout.write("GONIOTILTSTEP    %.6f\n" % stepsize)
            fout.write("BEAMTILTSTEP    %s\n" % beam_tilt_step)
            fout.write("BEAMTILTRANGE    %.3f\n" % beam_tilt_range)
            fout.write("STRETCHINGMP    %.1f\n" % stretching_mp)
            fout.write("STRETCHINGAZIMUTH    %.1f\n\n" % stretching_azimuth)

            fout.write("FILELIST\n")
            for idx, angle in enumerate(angles, start=1):
                fout.write(
                    "FILE %05d.mrc        %8.4f    0        %8.4f\n"
                    % (idx, angle, angle)
                )
            fout.write("ENDFILELIST\n")

    dst_root.mkdir(parents=True, exist_ok=True)
    with (dst_root / "delay_map.txt").open("w", encoding="utf-8") as f:
        for d in sorted_delays:
            f.write("%s\t%.3f\n" % (delay_to_exp[d], d))

    print("[3DED] Reconstruction prep finished: %s" % dst_root)
    return str(dst_root)


def angle_limit_exceeded(angle_list, limit=50.0):
    return any(abs(float(angle)) > limit for angle in angle_list)


def angle_preflight_signature(angle_list):
    return tuple(round(float(angle), 6) for angle in angle_list)


def preflight_angle_rotation(angle_list, limit=50.0, settle_time=0.2, should_stop=None):
    print(
        "[Safety] Angle exceeds +/-%.1f deg. "
        "Running angle-only preflight rotation before acquisition."
        % limit
    )

    for angle in angle_list:
        if should_stop is not None and should_stop():
            print("[Safety] Angle-only preflight rotation aborted.")
            return False

        print("[Safety] Preflight alpha tilt to %.4f deg" % angle)
        actual_alpha = tem_tilt(angle)
        print("[Safety] Preflight actual alpha = %.4f deg" % actual_alpha)
        time.sleep(settle_time)

    print("[Safety] Angle-only preflight rotation finished.")
    return True


class ApplicationWindow(QtWidgets.QMainWindow):
    tem_alpha_signal = QtCore.pyqtSignal(float)
    tem_status_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._angle_preflight_signature = None

        self._main = QtWidgets.QWidget()
        self.setWindowTitle("Pump-Probe 3DED Demo")
        self.setCentralWidget(self._main)

        layout = QtWidgets.QGridLayout(self._main)
        layout.setRowStretch(0, 4)

        label = QtWidgets.QLabel("Save to ", self._main)
        self._entry_dir = QtWidgets.QLineEdit("./", self._main)
        self._entry_dir.setReadOnly(True)

        self._btn_dir = QtWidgets.QPushButton("Choose", self._main)
        self._btn_dir.clicked.connect(self.choose_dir)

        layout.addWidget(label, 0, 0, 1, 1)
        layout.addWidget(self._entry_dir, 0, 1, 1, 2)
        layout.addWidget(self._btn_dir, 0, 3, 1, 1)

        onlyInt = QtGui.QIntValidator()
        onlyDouble = QtGui.QDoubleValidator()

        # Delay parameters
        layout.addWidget(QtWidgets.QLabel("Delay Start (ps)", self._main), 1, 0, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Delay Step (ps)", self._main), 2, 0, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Delay Stop (ps)", self._main), 3, 0, 1, 1)

        self._entry_start = QtWidgets.QLineEdit("1.0", self._main)
        self._entry_start.setValidator(onlyDouble)

        self._entry_step = QtWidgets.QLineEdit("0.1", self._main)
        self._entry_step.setValidator(onlyDouble)

        self._entry_stop = QtWidgets.QLineEdit("10.0", self._main)
        self._entry_stop.setValidator(onlyDouble)

        layout.addWidget(self._entry_start, 1, 1, 1, 1)
        layout.addWidget(self._entry_step, 2, 1, 1, 1)
        layout.addWidget(self._entry_stop, 3, 1, 1, 1)

        # Delay scan mode
        layout.addWidget(QtWidgets.QLabel("Mode ", self._main), 1, 2, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Loop ", self._main), 2, 2, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Order ", self._main), 3, 2, 1, 1)

        self._combo_mode = QtWidgets.QComboBox(self._main)
        self._combo_mode.addItem("Forward")
        self._combo_mode.addItem("Backward")
        self._combo_mode.addItem("Forward/Backward")

        self._entry_loop = QtWidgets.QLineEdit("2", self._main)
        self._entry_loop.setValidator(onlyInt)

        self._combo_order = QtWidgets.QComboBox(self._main)
        self._combo_order.addItem("Angle -> Delay")
        self._combo_order.addItem("Delay -> Angle")

        layout.addWidget(self._combo_mode, 1, 3, 1, 1)
        layout.addWidget(self._entry_loop, 2, 3, 1, 1)
        layout.addWidget(self._combo_order, 3, 3, 1, 1)

        # Angle parameters
        layout.addWidget(QtWidgets.QLabel("Angle Start (deg)", self._main), 4, 0, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Angle Step (deg)", self._main), 5, 0, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Angle Stop (deg)", self._main), 6, 0, 1, 1)

        self._entry_angle_start = QtWidgets.QLineEdit("-1.0", self._main)
        self._entry_angle_start.setValidator(onlyDouble)

        self._entry_angle_step = QtWidgets.QLineEdit("0.1", self._main)
        self._entry_angle_step.setValidator(onlyDouble)

        self._entry_angle_stop = QtWidgets.QLineEdit("1.0", self._main)
        self._entry_angle_stop.setValidator(onlyDouble)

        layout.addWidget(self._entry_angle_start, 4, 1, 1, 1)
        layout.addWidget(self._entry_angle_step, 5, 1, 1, 1)
        layout.addWidget(self._entry_angle_stop, 6, 1, 1, 1)

        self._check_reconstruct = QtWidgets.QCheckBox("to 3DED format", self._main)
        self._check_reconstruct.setChecked(True)
        layout.addWidget(self._check_reconstruct, 7, 0, 1, 2)

        # 3DED reconstruction parameters
        self._group_reconstruction = QtWidgets.QGroupBox("3DED format parameters", self._main)
        reconstruction_layout = QtWidgets.QGridLayout(self._group_reconstruction)

        self._entry_wavelength = QtWidgets.QLineEdit(CONFIG["WAVELENGTH"], self._main)
        self._entry_wavelength.setValidator(onlyDouble)

        self._entry_ccd_pixel_size = QtWidgets.QLineEdit(CONFIG["CCD_PIXEL_SIZE"], self._main)
        self._entry_ccd_pixel_size.setValidator(onlyDouble)

        self._entry_rotation_axis = QtWidgets.QLineEdit(CONFIG["ROTATION_AXIS"], self._main)
        self._entry_rotation_axis.setValidator(onlyDouble)

        self._entry_beam_tilt_step = QtWidgets.QLineEdit(CONFIG["BEAM_TILT_STEP"], self._main)
        self._entry_beam_tilt_step.setValidator(onlyDouble)

        self._entry_beam_tilt_range = QtWidgets.QLineEdit(CONFIG["BEAM_TILT_RANGE"], self._main)
        self._entry_beam_tilt_range.setValidator(onlyDouble)

        self._entry_stretching_mp = QtWidgets.QLineEdit(CONFIG["STRETCHING_MP"], self._main)
        self._entry_stretching_mp.setValidator(onlyDouble)

        self._entry_stretching_azimuth = QtWidgets.QLineEdit(CONFIG["STRETCHING_AZIMUTH"], self._main)
        self._entry_stretching_azimuth.setValidator(onlyDouble)

        reconstruction_layout.addWidget(QtWidgets.QLabel("Wavelength", self._group_reconstruction), 0, 0, 1, 1)
        reconstruction_layout.addWidget(self._entry_wavelength, 0, 1, 1, 1)
        reconstruction_layout.addWidget(QtWidgets.QLabel("Beam tilt step", self._group_reconstruction), 0, 2, 1, 1)
        reconstruction_layout.addWidget(self._entry_beam_tilt_step, 0, 3, 1, 1)

        reconstruction_layout.addWidget(QtWidgets.QLabel("CCD pixel size", self._group_reconstruction), 1, 0, 1, 1)
        reconstruction_layout.addWidget(self._entry_ccd_pixel_size, 1, 1, 1, 1)
        reconstruction_layout.addWidget(QtWidgets.QLabel("Beam tilt range", self._group_reconstruction), 1, 2, 1, 1)
        reconstruction_layout.addWidget(self._entry_beam_tilt_range, 1, 3, 1, 1)

        reconstruction_layout.addWidget(QtWidgets.QLabel("Rotation axis", self._group_reconstruction), 2, 0, 1, 1)
        reconstruction_layout.addWidget(self._entry_rotation_axis, 2, 1, 1, 1)
        reconstruction_layout.addWidget(QtWidgets.QLabel("Stretching MP", self._group_reconstruction), 2, 2, 1, 1)
        reconstruction_layout.addWidget(self._entry_stretching_mp, 2, 3, 1, 1)

        reconstruction_layout.addWidget(QtWidgets.QLabel("Stretching azimuth", self._group_reconstruction), 3, 2, 1, 1)
        reconstruction_layout.addWidget(self._entry_stretching_azimuth, 3, 3, 1, 1)

        layout.addWidget(self._group_reconstruction, 8, 0, 4, 4)

        # TEM manual control panel
        self._group_tem = QtWidgets.QGroupBox("TEM control", self._main)
        tem_layout = QtWidgets.QGridLayout(self._group_tem)

        self._label_current_alpha = QtWidgets.QLabel("--", self._group_tem)
        self._entry_tem_target_alpha = QtWidgets.QLineEdit("0.0", self._group_tem)
        self._entry_tem_target_alpha.setValidator(onlyDouble)

        self._btn_tem_get_alpha = QtWidgets.QPushButton("Get alpha", self._group_tem)
        self._btn_tem_tilt = QtWidgets.QPushButton("Tilt", self._group_tem)
        self._label_tem_status = QtWidgets.QLabel("Idle", self._group_tem)

        tem_layout.addWidget(QtWidgets.QLabel("Current alpha (deg)", self._group_tem), 0, 0, 1, 1)
        tem_layout.addWidget(self._label_current_alpha, 0, 1, 1, 1)
        tem_layout.addWidget(self._btn_tem_get_alpha, 0, 2, 1, 1)

        tem_layout.addWidget(QtWidgets.QLabel("Target alpha (deg)", self._group_tem), 1, 0, 1, 1)
        tem_layout.addWidget(self._entry_tem_target_alpha, 1, 1, 1, 1)
        tem_layout.addWidget(self._btn_tem_tilt, 1, 2, 1, 1)

        tem_layout.addWidget(QtWidgets.QLabel("Status", self._group_tem), 2, 0, 1, 1)
        tem_layout.addWidget(self._label_tem_status, 2, 1, 1, 2)

        layout.addWidget(self._group_tem, 12, 0, 3, 4)

        self.tem_alpha_signal.connect(self.update_tem_alpha)
        self.tem_status_signal.connect(self.update_tem_status)
        self._btn_tem_get_alpha.clicked.connect(self.get_tem_alpha)
        self._btn_tem_tilt.clicked.connect(self.tilt_tem_alpha)

        self._btn_run = QtWidgets.QPushButton(self._main)
        self._btn_run.setText("Run")
        layout.addWidget(self._btn_run, 4, 2, 3, 2)
        self._btn_run.clicked.connect(self.run)

    @QtCore.pyqtSlot(float)
    def update_tem_alpha(self, alpha):
        self._label_current_alpha.setText("%.4f" % alpha)

    @QtCore.pyqtSlot(str)
    def update_tem_status(self, status):
        self._label_tem_status.setText(status)

    @QtCore.pyqtSlot()
    def get_tem_alpha(self):
        if self._btn_run.text() == "Abort":
            self.tem_status_signal.emit("TEM control disabled during acquisition.")
            return

        self.tem_status_signal.emit("Reading alpha...")
        thread = threading.Thread(target=self.get_tem_alpha_worker)
        thread.daemon = True
        thread.start()

    def get_tem_alpha_worker(self):
        try:
            alpha = tem_get_alpha()
            self.tem_alpha_signal.emit(alpha)
            self.tem_status_signal.emit("Current alpha = %.4f deg" % alpha)
        except Exception as exc:
            self.tem_status_signal.emit("GET_ALPHA failed: %s" % exc)

    @QtCore.pyqtSlot()
    def tilt_tem_alpha(self):
        if self._btn_run.text() == "Abort":
            self.tem_status_signal.emit("TEM control disabled during acquisition.")
            return

        target_alpha = float(self._entry_tem_target_alpha.text())
        self.tem_status_signal.emit("Tilting to %.4f deg..." % target_alpha)
        thread = threading.Thread(target=self.tilt_tem_alpha_worker, args=(target_alpha,))
        thread.daemon = True
        thread.start()

    def tilt_tem_alpha_worker(self, target_alpha):
        try:
            actual_alpha = tem_tilt(target_alpha)
            self.tem_alpha_signal.emit(actual_alpha)
            self.tem_status_signal.emit("Tilt done: %.4f deg" % actual_alpha)
        except Exception as exc:
            self.tem_status_signal.emit("TILT failed: %s" % exc)

    @QtCore.pyqtSlot()
    def choose_dir(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog

        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Directory", ".", options=options
        )

        self._entry_dir.setText(folder_path)

    def confirm_acquisition_parameters(
        self, start, step, stop, mode, loop, acquisition_order,
        angle_start, angle_step, angle_stop, run_reconstruction,
        reconstruction_params
    ):
        def angle_text(value):
            text = "%.4f deg" % value
            if abs(value) > 50:
                return '<span style="color: red; font-weight: bold;">%s</span>' % text
            return text

        angle_warning = ""
        if abs(angle_start) > 50 or abs(angle_stop) > 50:
            angle_warning = (
                '<p style="color: red; font-weight: bold;">'
                'Warning: alpha start or stop exceeds +/-50 deg. '
                'The first Run will perform angle-only preflight; '
                'click Run again after preflight to start acquisition.'
                '</p>'
            )

        mode_text = self._combo_mode.itemText(mode)
        order_text = self._combo_order.itemText(acquisition_order)
        tilt_range = angle_stop - angle_start
        delay_range = stop - start

        message = (
            "<h3>Confirm acquisition parameters</h3>"
            "<table cellspacing='6'>"
            "<tr><td><b>Angle start</b></td><td>%s</td></tr>"
            "<tr><td><b>Angle stop</b></td><td>%s</td></tr>"
            "<tr><td><b>Tilt range</b></td><td>%.4f deg</td></tr>"
            "<tr><td><b>Angle step</b></td><td>%.4f deg</td></tr>"
            "<tr><td><b>Delay start</b></td><td>%.4f ps</td></tr>"
            "<tr><td><b>Delay stop</b></td><td>%.4f ps</td></tr>"
            "<tr><td><b>Time-delay range</b></td><td>%.4f ps</td></tr>"
            "<tr><td><b>Delay step</b></td><td>%.4f ps</td></tr>"
            "<tr><td><b>Mode</b></td><td>%s</td></tr>"
            "<tr><td><b>Order</b></td><td>%s</td></tr>"
            "<tr><td><b>Loop</b></td><td>%d</td></tr>"
            "<tr><td><b>to 3DED format</b></td><td>%s</td></tr>"
            "<tr><td colspan='2'><b>3DED format parameters</b></td></tr>"
            "<tr><td><b>Wavelength</b></td><td>%.6f</td></tr>"
            "<tr><td><b>CCD pixel size</b></td><td>%.6f</td></tr>"
            "<tr><td><b>Rotation axis</b></td><td>%.6f</td></tr>"
            "<tr><td><b>Beam tilt step</b></td><td>%.6f</td></tr>"
            "<tr><td><b>Beam tilt range</b></td><td>%.6f</td></tr>"
            "<tr><td><b>Stretching MP</b></td><td>%.6f</td></tr>"
            "<tr><td><b>Stretching azimuth</b></td><td>%.6f</td></tr>"
            "</table>"
            "%s"
            "<p>Start acquisition?</p>"
            % (
                angle_text(angle_start),
                angle_text(angle_stop),
                tilt_range,
                angle_step,
                start,
                stop,
                delay_range,
                step,
                mode_text,
                order_text,
                loop,
                "Yes" if run_reconstruction else "No",
                reconstruction_params["wavelength"],
                reconstruction_params["ccd_pixel_size"],
                reconstruction_params["rotation_axis"],
                reconstruction_params["beam_tilt_step"],
                reconstruction_params["beam_tilt_range"],
                reconstruction_params["stretching_mp"],
                reconstruction_params["stretching_azimuth"],
                angle_warning,
            )
        )

        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Confirm Parameters")
        box.setTextFormat(QtCore.Qt.RichText)
        box.setIcon(QtWidgets.QMessageBox.Warning if angle_warning else QtWidgets.QMessageBox.Question)
        box.setText(message)
        box.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        box.setDefaultButton(QtWidgets.QMessageBox.Cancel if angle_warning else QtWidgets.QMessageBox.Ok)

        return box.exec_() == QtWidgets.QMessageBox.Ok

    @QtCore.pyqtSlot()
    def run(self):
        curr_state = self._btn_run.text()

        if curr_state == "Run":
            directory = self._entry_dir.text()

            start = float(self._entry_start.text())
            step = float(self._entry_step.text())
            stop = float(self._entry_stop.text())

            mode = self._combo_mode.currentIndex()
            loop = int(self._entry_loop.text())
            acquisition_order = self._combo_order.currentIndex()

            angle_start = float(self._entry_angle_start.text())
            angle_step = float(self._entry_angle_step.text())
            angle_stop = float(self._entry_angle_stop.text())
            run_reconstruction = self._check_reconstruct.isChecked()
            reconstruction_params = {
                "wavelength": float(self._entry_wavelength.text()),
                "ccd_pixel_size": float(self._entry_ccd_pixel_size.text()),
                "rotation_axis": float(self._entry_rotation_axis.text()),
                "beam_tilt_step": float(self._entry_beam_tilt_step.text()),
                "beam_tilt_range": float(self._entry_beam_tilt_range.text()),
                "stretching_mp": float(self._entry_stretching_mp.text()),
                "stretching_azimuth": float(self._entry_stretching_azimuth.text()),
            }

            if not self.confirm_acquisition_parameters(
                start, step, stop, mode, loop, acquisition_order,
                angle_start, angle_step, angle_stop, run_reconstruction,
                reconstruction_params
            ):
                return

            self._btn_run.setText("Abort")
            self.stop_flag = False

            thread0 = threading.Thread(
                target=self.real_process_function,
                args=(
                    directory, start, step, stop, mode, loop,
                    angle_start, angle_step, angle_stop,
                    acquisition_order, run_reconstruction,
                    reconstruction_params
                )
            )
            thread0.start()

        else:
            self.stop_flag = True

    def real_process_function(
        self, directory, start, step, stop, mode, loop,
        angle_start, angle_step, angle_stop, acquisition_order,
        run_reconstruction, reconstruction_params
    ):
        print(
            "Enter process function with parameters: "
            "dir=%s, delay_start=%s, delay_step=%s, delay_stop=%s, "
            "mode=%d, loop=%d, angle_start=%s, angle_step=%s, angle_stop=%s, "
            "acquisition_order=%d, run_reconstruction=%s, reconstruction_params=%s"
            % (
                directory, start, step, stop, mode, loop,
                angle_start, angle_step, angle_stop, acquisition_order,
                run_reconstruction, reconstruction_params
            )
        )

        try:
            self.my_func(
                directory, start, step, stop, mode, loop,
                angle_start, angle_step, angle_stop, acquisition_order,
                run_reconstruction, reconstruction_params
            )

        except Exception:
            traceback.print_exc()

        finally:
            self._btn_run.setText("Run")

    def my_func(
        self, directory, start, step, stop, mode, loop,
        angle_start, angle_step, angle_stop, acquisition_order,
        run_reconstruction, reconstruction_params
    ):

        serverurl = SERVAL_URL

        check_connection(serverurl)

        dashboard = get_dashboard(serverurl)
        print('Server Software Version:', dashboard['Server']['SoftwareVersion'])
        print('Dashboard:', dashboard)

        bpcFile = BPC_FILE
        dacsFile = DACS_FILE

        init_cam(serverurl, bpcFile, dacsFile)

        response = get_request(url=serverurl + '/detector/config')
        data = response.text
        print('Response of getting the Detector Configuration from SERVAL: ' + data)

        detectorConfig = json.loads(data)
        detectorConfig["BiasVoltage"] = 100
        detectorConfig["BiasEnabled"] = True

        detectorconfig = get_detectorconfig(serverurl)

        init_acquisition(
            serverurl,
            detectorConfig,
            detectorconfig['nTriggers'],
            detectorconfig['TriggerPeriod'],
            detectorconfig['ExposureTime']
        )

        response = get_request(url=serverurl + '/detector/config')
        print('Response of getting the updated Detector Configuration from SERVAL : ' + response.text)

        os.chdir(WORK_DIR)
        print('current dirrectory: ' + WORK_DIR)

        start_time = start
        stop_time = stop
        step_time = step

        travel_limit_positive = 392.000
        travel_limit_negative = 5.000

        start_position = travel_limit_positive - 2.998 * start_time * 0.1 / 2
        stop_position = travel_limit_positive - 2.998 * stop_time * 0.1 / 2
        step_distance = 2.998 * step_time * 0.1 / 2

        num_step = int((stop_time - start_time) / step_time + 1)

        # Generate angle list
        if angle_step == 0:
            raise ValueError("Angle step cannot be 0.")

        if angle_stop >= angle_start and angle_step < 0:
            raise ValueError("Angle step must be positive when angle_stop >= angle_start.")

        if angle_stop < angle_start and angle_step > 0:
            raise ValueError("Angle step must be negative when angle_stop < angle_start.")

        angle_list = np.arange(
            angle_start,
            angle_stop + angle_step / 2,
            angle_step
        )

        base_directory = directory

        print("[3DED] Angle list:", angle_list)

        if angle_limit_exceeded(angle_list):
            preflight_signature = angle_preflight_signature(angle_list)

            if self._angle_preflight_signature != preflight_signature:
                preflight_ok = preflight_angle_rotation(
                    angle_list,
                    should_stop=lambda: self.stop_flag
                )
                if preflight_ok:
                    self._angle_preflight_signature = preflight_signature
                    print(
                        "[Safety] Preflight completed. "
                        "Please click Run again to start data acquisition."
                    )
                return

            print("[Safety] Matching angle preflight already completed; starting acquisition.")

            if self.stop_flag:
                return

        def delay_to_position(delay_value):
            return travel_limit_positive - 2.998 * delay_value * 0.1 / 2

        def delay_values_for_loop(loop_index):
            if mode == 0:
                return [start_time + i * step_time for i in range(num_step)]
            if mode == 1:
                return [stop_time - i * step_time for i in range(num_step)]
            if (loop_index % 2) == 0:
                return [start_time + i * step_time for i in range(num_step)]
            return [stop_time - i * step_time for i in range(num_step)]

        if acquisition_order == 1:
            print("[Order] Delay -> Angle")

            for j in range(loop):
                if self.stop_flag:
                    return

                delay_values = delay_values_for_loop(j)
                print(f"[Loop] {j + 1}/{loop}, mode {mode}, delay-first acquisition")

                for delay_value in delay_values:
                    if self.stop_flag:
                        return

                    delay_position = delay_to_position(delay_value)
                    print(f"[Delay] Moving delay stage to {delay_value} ps ({delay_position:.4f} mm)")
                    stage.move_to(delay_position, True)
                    time.sleep(DELAY_STAGE_SETTLE_TIME)

                    for angle in angle_list:
                        if self.stop_flag:
                            return

                        angle_dir_name = f"angle_{angle:+07.2f}"
                        angle_directory = os.path.join(base_directory, angle_dir_name)
                        os.makedirs(angle_directory, exist_ok=True)

                        print(f"[TEM] Moving alpha tilt to {angle:.4f} deg")
                        actual_alpha = tem_tilt(angle)
                        print(f"[TEM] Current alpha = {actual_alpha:.4f} deg")
                        time.sleep(TEM_TILT_SETTLE_TIME)
                        print(f"[Data] Saving this angle to: {angle_directory}")

                        destination = {
                            "Preview": {
                                "SamplingMode": "skipOnFrame",
                                "Period": 0.2,
                                "ImageChannels": [{
                                    "Base": pathlib.Path(angle_directory).as_uri(),
                                    "FilePattern": "aaa%Hms_",
                                    "Format": "tiff",
                                    "Mode": "count"
                                }, {
                                    "Base": "http://localhost",
                                    "Format": "png",
                                    "Mode": "count"
                                }]
                            }
                        }

                        response = put_request(
                            url=serverurl + '/server/destination',
                            data=json.dumps(destination)
                        )
                        print('Response of uploading the Destination Configuration to SERVAL : ' + response.text)

                        response = get_request(url=serverurl + '/server/destination')
                        print('Selected destination : ' + response.text)

                        print(f"[Acquire] angle={angle:.4f}, delay={delay_value} ps, loop={j + 1}")
                        simple_acquisition(serverurl)

                        renamed_files = file_rename(
                            angle_directory,
                            delay_value,
                            j + 1,
                            detectorconfig['nTriggers']
                        )
                        append_acquisition_log(
                            angle_directory, angle, actual_alpha,
                            delay_value, j + 1, renamed_files
                        )

                        time.sleep(0.1)

            if run_reconstruction:
                print("[3DED] Preparing reconstruction input files...")
                build_3ded_reconstruction(base_directory, **reconstruction_params)

            print("Experiment is done!")
            return

        print("[Order] Angle -> Delay")

        for angle in angle_list:
            if self.stop_flag:
                return

            angle_dir_name = f"angle_{angle:+07.2f}"
            angle_directory = os.path.join(base_directory, angle_dir_name)
            os.makedirs(angle_directory, exist_ok=True)

            print(f"[TEM] Moving alpha tilt to {angle:.4f} deg")
            actual_alpha = tem_tilt(angle)
            print(f"[TEM] Current alpha = {actual_alpha:.4f} deg")
            time.sleep(TEM_TILT_SETTLE_TIME)
            print(f"[Data] Saving this angle to: {angle_directory}")

            destination = {
                "Preview": {
                    "SamplingMode": "skipOnFrame",
                    "Period": 0.2,
                    "ImageChannels": [{
                        "Base": pathlib.Path(angle_directory).as_uri(),
                        "FilePattern": "aaa%Hms_",
                        "Format": "tiff",
                        "Mode": "count"
                    }, {
                        "Base": "http://localhost",
                        "Format": "png",
                        "Mode": "count"
                    }]
                }
            }

            response = put_request(
                url=serverurl + '/server/destination',
                data=json.dumps(destination)
            )
            print('Response of uploading the Destination Configuration to SERVAL : ' + response.text)

            response = get_request(url=serverurl + '/server/destination')
            print('Selected destination : ' + response.text)

            # ============================
            # Original delay scan starts
            # ============================
            for j in range(loop):
                if self.stop_flag:
                    return

                print(f"[Angle] {angle:.4f} deg, loop {j + 1}/{loop}, mode {mode}")

                if mode == 0:
                    stage.move_to(start_position, True)
                    time.sleep(DELAY_STAGE_SETTLE_TIME)

                    for i in range(num_step):
                        if not stage.moving:
                            delay_value = start_time + i * step_time

                            print(f"[Acquire] angle={angle:.4f}, delay={delay_value} ps, loop={j + 1}")
                            simple_acquisition(serverurl)

                            renamed_files = file_rename(
                                angle_directory,
                                delay_value,
                                j + 1,
                                detectorconfig['nTriggers']
                            )
                            append_acquisition_log(
                                angle_directory, angle, actual_alpha,
                                delay_value, j + 1, renamed_files
                            )

                            time.sleep(0.1)

                            t = start_position - step_distance * (i + 1)
                            stage.move_to(t, True)

                            print("Time delay position (ps): %s" % (start_time + step_time * (i + 1)))
                            time.sleep(DELAY_STAGE_SETTLE_TIME)

                        if self.stop_flag:
                            return

                elif mode == 1:
                    stage.move_to(stop_position, True)
                    time.sleep(DELAY_STAGE_SETTLE_TIME)

                    for i in range(num_step):
                        if not stage.moving:
                            delay_value = stop_time - i * step_time

                            print(f"[Acquire] angle={angle:.4f}, delay={delay_value} ps, loop={j + 1}")
                            simple_acquisition(serverurl)

                            renamed_files = file_rename(
                                angle_directory,
                                delay_value,
                                j + 1,
                                detectorconfig['nTriggers']
                            )
                            append_acquisition_log(
                                angle_directory, angle, actual_alpha,
                                delay_value, j + 1, renamed_files
                            )

                            time.sleep(0.1)

                            stage.move_to(stop_position + step_distance * (i + 1), True)

                            print("Time delay position (ps): %s" % (stop_time - step_time * (i + 1)))
                            time.sleep(DELAY_STAGE_SETTLE_TIME)

                        if self.stop_flag:
                            return

                else:
                    if (j % 2) == 0:
                        stage.move_to(start_position, True)
                        time.sleep(DELAY_STAGE_SETTLE_TIME)

                        for i in range(num_step):
                            if not stage.moving:
                                delay_value = start_time + i * step_time

                                print(f"[Acquire] angle={angle:.4f}, delay={delay_value} ps, loop={j + 1}")
                                simple_acquisition(serverurl)

                                renamed_files = file_rename(
                                    angle_directory,
                                    delay_value,
                                    j + 1,
                                    detectorconfig['nTriggers']
                                )
                                append_acquisition_log(
                                    angle_directory, angle, actual_alpha,
                                    delay_value, j + 1, renamed_files
                                )

                                time.sleep(0.1)

                                stage.move_to(start_position - step_distance * (i + 1), True)

                                print("Time delay position (ps): %s" % (start_time + step_time * (i + 1)))
                                time.sleep(DELAY_STAGE_SETTLE_TIME)

                            if self.stop_flag:
                                return

                    else:
                        stage.move_to(stop_position, True)
                        time.sleep(DELAY_STAGE_SETTLE_TIME)

                        for i in range(num_step):
                            if not stage.moving:
                                delay_value = stop_time - i * step_time

                                print(f"[Acquire] angle={angle:.4f}, delay={delay_value} ps, loop={j + 1}")
                                simple_acquisition(serverurl)

                                renamed_files = file_rename(
                                    angle_directory,
                                    delay_value,
                                    j + 1,
                                    detectorconfig['nTriggers']
                                )
                                append_acquisition_log(
                                    angle_directory, angle, actual_alpha,
                                    delay_value, j + 1, renamed_files
                                )

                                time.sleep(0.1)

                                stage.move_to(stop_position + step_distance * (i + 1), True)

                                print("Time delay position (ps): %s" % (stop_time - step_time * (i + 1)))
                                time.sleep(DELAY_STAGE_SETTLE_TIME)

                            if self.stop_flag:
                                return

        if run_reconstruction:
            print("[3DED] Preparing reconstruction input files...")
            build_3ded_reconstruction(base_directory, **reconstruction_params)

        print("Experiment is done!")


if __name__ == '__main__':
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    qapp.exec_()
