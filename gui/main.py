import os
for var in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
    if var in os.environ:
        os.environ.pop(var)
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

import sys, json, time
import numpy as np
import cv2
from collections import deque
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QSignalBlocker, QPoint, QMutex, QMutexLocker
from PyQt6.QtGui import QImage, QPixmap, QAction, QKeySequence, QShortcut, QPainter, QPen, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QHBoxLayout, QVBoxLayout, QGroupBox, QCheckBox, QTextEdit, QMessageBox,
    QGridLayout, QSlider, QToolBar, QFileDialog, QMenu
)
from hazmat import HazmatYOLO
from qrscan import QRScanner
from esp32cam import CameraWorker, ESP32CamClient, extract_host_from_stream_url, ParameterSetterThread
from magnetometer import MagnetometerWorker

def np_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)

class DragAwareSlider(QSlider):
    dragStarted = pyqtSignal()
    dragEnded   = pyqtSignal()

    def mousePressEvent(self, e):
        self.dragStarted.emit()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        self.dragEnded.emit()


class FileIOWorker(QThread):
    finished = pyqtSignal(bool, str)
    
    def __init__(self, operation: str, path: str, data=None):
        super().__init__()
        self.operation = operation
        self.path = path
        self.data = data
        self.result = None
        
    def run(self):
        try:
            if self.operation == 'save':
                with open(self.path, "w") as f:
                    json.dump(self.data, f, indent=2)
                self.finished.emit(True, self.path)
            elif self.operation == 'load':
                with open(self.path, "r") as f:
                    self.result = json.load(f)
                self.finished.emit(True, self.path)
        except Exception as e:
            self.finished.emit(False, str(e))


class CameraPanel(QGroupBox):
    def __init__(self, title: str, url: str):
        super().__init__(title)
        self.url = url
        self.worker: CameraWorker | None = None
        self.client: ESP32CamClient | None = ESP32CamClient(url)
        
        self._state_mutex = QMutex()
        self._last_frame_ts = time.monotonic()
        self._restart_in_progress = False
        
        self._param_setter: ParameterSetterThread | None = None
        self._file_worker: FileIOWorker | None = None

        self.url_edit = QLineEdit(url)
        self.url_edit.setStyleSheet(
            "font-size: 11px; font-family: monospace; color:#9efc6a; "
            "background:#1b1b1d; border:1px solid #555; border-radius:6px;"
        )
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop"); self.btn_stop.setEnabled(False)

        self.chk_haz = QCheckBox("Hazmat")
        self.chk_qr = QCheckBox("QR")
        self.chk_haz.setStyleSheet("color: #9efc6a; font-weight: bold;")
        self.chk_qr.setStyleSheet("color: #ffd54f; font-weight: bold;")
        self.fps_label = QLabel("FPS: —")

        self.status_dot = QLabel("● disconnected")
        self.status_dot.setStyleSheet("color:#e53935; font-weight:bold;")

        self.fps_bar = QLabel()
        self.fps_bar.setFixedHeight(8)
        self.fps_bar.setFixedWidth(6)
        self.fps_bar.setStyleSheet("background:#333; border-radius:4px;")

        self.hazmat_box = QTextEdit(); self.hazmat_box.setReadOnly(True)
        self.hazmat_box.setStyleSheet("background:#222; color:#0f0; border-radius:6px;")
        self.hazmat_box.setPlaceholderText("Hazmat detections...")

        self.qr_box = QTextEdit(); self.qr_box.setReadOnly(True)
        self.qr_box.setStyleSheet("background:#222; color:#ff0; border-radius:6px;")
        self.qr_box.setPlaceholderText("QR detections...")

        self.view = QLabel("No video")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setStyleSheet("background:#000; color:#aaa; border:1px solid #333; border-radius:10px;")
        self.view.setMinimumSize(800, 450)
        
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._show_view_menu)

        left = QVBoxLayout()
        left.addWidget(QLabel("Stream URL:"))
        left.addWidget(self.url_edit)
        row = QHBoxLayout()
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        left.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(self.chk_haz)
        row2.addWidget(self.chk_qr)
        left.addLayout(row2)
        left.addWidget(self.fps_label)
        left.addWidget(self.status_dot)
        left.addWidget(self.fps_bar)

        left.addSpacing(6)
        left.addWidget(QLabel("— Tuning —"))

        checkbox_style = """
            QCheckBox {
                color: white;
                font-weight: bold;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #aaa;
                background-color: #2b2b2e;
            }
            QCheckBox::indicator:checked {
                background-color: #00c853;
                border: 1px solid #00e676;
            }
            QCheckBox::indicator:unchecked {
                background-color: #1b1b1d;
                border: 1px solid #555;
            }
        """

        self._tune_debounce = QTimer(self)
        self._tune_debounce.setSingleShot(True)
        self._tune_debounce.setInterval(320)
        self._last_tune_cb = None

        def schedule(cb):
            self._tune_debounce.stop()
            self._last_tune_cb = cb
            self._tune_debounce.start()

        def on_debounce():
            try:
                if self._last_tune_cb:
                    self._last_tune_cb()
            except Exception as e:
                print("[tune-debounce]", e)

        self._tune_debounce.timeout.connect(on_debounce)

        def add_slider(caption, minv, maxv, init, apply_fn, klass=QSlider):
            row = QHBoxLayout()
            lab = QLabel(caption)
            lab.setStyleSheet("color:white;")
            row.addWidget(lab)
            s = klass(Qt.Orientation.Horizontal)
            s.setMinimum(minv)
            s.setMaximum(maxv)
            s.setValue(init)
            s.setFixedWidth(160)
            s.valueChanged.connect(lambda _v: schedule(lambda: apply_fn(s.value())))
            row.addWidget(s)
            left.addLayout(row)
            return s

        def add_check(caption, init, apply_fn):
            c = QCheckBox(caption); c.setChecked(init)
            c.setStyleSheet(checkbox_style)
            c.stateChanged.connect(lambda _v: apply_fn(1 if c.isChecked() else 0))
            left.addWidget(c)
            return c

        self.sl_bright = add_slider("Brightness", -2, 2, 0,  lambda v: self._set_param(brightness=v))
        self.sl_contr = add_slider("Contrast",   -2, 2, 0,  lambda v: self._set_param(contrast=v))
        self.sl_satur = add_slider("Saturation", -2, 2, 0,  lambda v: self._set_param(saturation=v))

        self.cb_awb = add_check("Auto WB", True,  lambda v: self._set_param(awb=v))
        self.cb_agc = add_check("Auto Gain", True,  lambda v: self._set_param(agc=v))
        self.cb_aec = add_check("Auto Exposure", True, lambda v: self._set_param(aec=v))

        self.sl_ae_level = add_slider("AE Level", -2, 2, 0, lambda v: self._set_param(ae_level=v) if self.cb_aec.isChecked() else None)

        self.sl_exp  = add_slider("Exposure", 0, 1200, 300, lambda v: self._set_param(aec_value=v), klass=DragAwareSlider)
        self.sl_gain = add_slider("Gain", 0, 30, 8, lambda v: self._set_param(agc_gain=v), klass=DragAwareSlider)
        self.sl_exp.dragStarted.connect(lambda: (self.cb_aec.setChecked(False), self._set_param(aec=0)))
        self.sl_gain.dragStarted.connect(lambda: (self.cb_agc.setChecked(False), self._set_param(agc=0)))

        row_orient = QHBoxLayout()
        self.cb_hm = QCheckBox("Mirror")
        self.cb_hm.setStyleSheet(checkbox_style)
        self.cb_vf = QCheckBox("Flip")
        self.cb_vf.setStyleSheet(checkbox_style)
        self.cb_hm.stateChanged.connect(self._on_flip_toggle)
        self.cb_vf.stateChanged.connect(self._on_flip_toggle)
        row_orient.addWidget(self.cb_hm)
        row_orient.addWidget(self.cb_vf)
        left.addLayout(row_orient)

        rowp = QHBoxLayout()
        btn_low = QPushButton("Preset: Low Light")
        btn_day = QPushButton("Preset: Daylight")
        btn_low.clicked.connect(lambda: self._apply_preset(aec=0, agc=1, awb=1, brightness=1, contrast=1, saturation=1, ae_level=0, aec_value=600, agc_gain=8))
        btn_day.clicked.connect(lambda: self._apply_preset(aec=1, agc=1, awb=1, brightness=0, contrast=0, saturation=0,ae_level=0, aec_value=300, agc_gain=8))
        rowp.addWidget(btn_low)
        rowp.addWidget(btn_day)
        left.addLayout(rowp)

        rowp2 = QHBoxLayout()
        btn_save = QPushButton("Save Profile")
        btn_load = QPushButton("Load Profile")
        btn_save.clicked.connect(self.save_profile)
        btn_load.clicked.connect(self.load_profile)
        rowp2.addWidget(btn_save)
        rowp2.addWidget(btn_load)
        left.addLayout(rowp2)

        left.addWidget(QLabel("Hazmat:"))
        left.addWidget(self.hazmat_box)
        left.addWidget(QLabel("QR:"))
        left.addWidget(self.qr_box)
        left.addStretch(1)

        root = QHBoxLayout()
        root.addLayout(left, 0)
        root.addWidget(self.view, 1)
        self.setLayout(root)

        self.btn_start.clicked.connect(self.start_stream)
        self.btn_stop.clicked.connect(self.stop_stream)
        self.chk_haz.stateChanged.connect(self.update_toggles)
        self.chk_qr.stateChanged.connect(self.update_toggles)

        self._watchdog = QTimer(self)
        self._watchdog.setInterval(600) 
        self._watchdog.timeout.connect(self._check_watchdog)
        self._watchdog.start()

    def _get_restart_in_progress(self) -> bool:
        with QMutexLocker(self._state_mutex):
            return self._restart_in_progress
    
    def _set_restart_in_progress(self, value: bool):
        with QMutexLocker(self._state_mutex):
            self._restart_in_progress = value
    
    def _update_frame_timestamp(self):
        with QMutexLocker(self._state_mutex):
            self._last_frame_ts = time.monotonic()
    
    def _get_time_since_last_frame(self) -> float:
        with QMutexLocker(self._state_mutex):
            return time.monotonic() - self._last_frame_ts

    def _set_param(self, **params):
        url = self.url_edit.text().strip()
        if not url:
            return

        if self._param_setter and self._param_setter.isRunning():
            self._param_setter.wait(150)
            if self._param_setter.isRunning():
                try:
                    self._param_setter.terminate()
                except Exception:
                    pass

                self._param_setter.wait(150)

        host = extract_host_from_stream_url(url)
        self._set_status("● tuning…", "#fdd835")

        self._param_setter = ParameterSetterThread(host, params)
        self._param_setter.finished.connect(self._on_param_set_finished)
        self._param_setter.start()

    def _on_param_set_finished(self, success: bool):
        QTimer.singleShot(200, self._set_status_connected_if_running)
        if self._param_setter:
            self._param_setter.deleteLater()
            self._param_setter = None

    def _apply_preset(self, *, aec, agc, awb, brightness, contrast, saturation, ae_level, aec_value, agc_gain):
        with QSignalBlocker(self.cb_aec), QSignalBlocker(self.cb_agc), QSignalBlocker(self.cb_awb):
            self.cb_aec.setChecked(bool(aec))
            self.cb_agc.setChecked(bool(agc))
            self.cb_awb.setChecked(bool(awb))
        
        for s, val in ((self.sl_bright, brightness), (self.sl_contr, contrast), (self.sl_satur, saturation), (self.sl_ae_level, ae_level), (self.sl_exp, aec_value), (self.sl_gain, agc_gain)):
            with QSignalBlocker(s):
                s.setValue(int(val))

        self._set_param(aec=int(aec), agc=int(agc), awb=int(awb))

        def phase2():
            payload = dict(brightness=int(brightness), contrast=int(contrast), saturation=int(saturation))
            if aec:
                payload["ae_level"] = int(ae_level)
            else: 
                payload["aec_value"] = int(aec_value)

            if not agc: 
                payload["agc_gain"] = int(agc_gain)

            self._set_param(**payload)

        QTimer.singleShot(140, phase2)

    def start_stream(self):
        if self.worker and self.worker.isRunning():
            return
        
        url = self.url_edit.text().strip()
        if not url:
            QMessageBox.warning(self, "URL missing", "Enter a stream URL.")
            return

        haz_factory = (lambda: HazmatYOLO("models/yolo.cfg", "models/yolo.weights", "models/labels.names", (416, 416), 0.8, 0.4, "opencv")) if HazmatYOLO else None
        qr_factory = (lambda: QRScanner()) if QRScanner else None

        self.worker = CameraWorker(url, hazmat_factory=haz_factory, qr_factory=qr_factory, target_fps=30.0)
        self.worker.enable_hazmat = self.chk_haz.isChecked() and (haz_factory is not None)
        self.worker.enable_qr = self.chk_qr.isChecked() and (qr_factory is not None)

        self.worker.frameReady.connect(self.on_frame, Qt.ConnectionType.QueuedConnection)
        self.worker.fpsReady.connect(self.on_fps, Qt.ConnectionType.QueuedConnection)
        self.worker.opened.connect(self.on_opened, Qt.ConnectionType.QueuedConnection)
        self.worker.hazmatDet.connect(self.on_hazmat, Qt.ConnectionType.QueuedConnection)
        self.worker.qrDet.connect(self.on_qr, Qt.ConnectionType.QueuedConnection)

        self._set_status("● connecting…", "#fdd835")
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_stream(self):
        if self.worker:
            self.worker.stop()
            self.worker = None

        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.fps_label.setText("Stopped")
        self.view.setText("No video")
        self.hazmat_box.clear()
        self.qr_box.clear()
        self.hazmat_box.setStyleSheet("background:#222; color:#0f0; border-radius:6px;")
        self.qr_box.setStyleSheet("background:#222; color:#ff0; border-radius:6px;")
        self._set_status("● disconnected", "#e53935")

    def update_toggles(self):
        if self.worker:
            self.worker.enable_hazmat = self.chk_haz.isChecked() and (self.worker.hazmat_factory is not None)
            self.worker.enable_qr = self.chk_qr.isChecked() and (self.worker.qr_factory is not None)

    def _set_status(self, txt: str, color_hex: str):
        self.status_dot.setText(txt)
        self.status_dot.setStyleSheet(f"color:{color_hex}; font-weight:bold;")

    def _set_status_connected_if_running(self):
        if self.worker and self.worker.isRunning():
            self._set_status("● connected", "#43a047")

    def _check_watchdog(self):
        if not (self.worker and self.worker.isRunning()):
            return
        
        dt = self._get_time_since_last_frame()
        if dt > 2.5 and not self._get_restart_in_progress():
            self._set_restart_in_progress(True)
            self._set_status("● reconnecting…", "#fdd835")
            self.stop_stream()
            QTimer.singleShot(400, self._do_restart)

    def _do_restart(self):
        self.start_stream()
        self._set_restart_in_progress(False)

    def on_opened(self, ok: bool, msg: str):
        if not ok:
            self._set_status("● disconnected", "#e53935")
            self.stop_stream()
            QMessageBox.critical(self, "Open error", msg)
        else:
            self._set_status("● connected", "#43a047")

    def _on_flip_toggle(self, _state=None):
        if hasattr(self, "_last_frame") and self._last_frame is not None:
            self.on_frame(self._last_frame)

    def on_frame(self, frame_bgr: np.ndarray):
        self._update_frame_timestamp()
        self._last_frame = frame_bgr.copy()

        try:
            hm = self.cb_hm.isChecked()
            vf = self.cb_vf.isChecked()
            if hm and vf:
                frame_bgr = cv2.flip(frame_bgr, -1)
            elif hm:
                frame_bgr = cv2.flip(frame_bgr, 1)
            elif vf:
                frame_bgr = cv2.flip(frame_bgr, 0)
        except Exception as e:
            print("[on_frame flip error]", e)

        pix = np_to_qpixmap(frame_bgr)
        scaled = pix.scaled(self.view.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.view.setPixmap(scaled)

    def on_fps(self, fps: float):
        self.fps_label.setText(f"FPS: {fps:.1f}")
        pct = max(0.0, min(1.0, fps / 30.0))
        w = int(200 * pct)
        color = "#e53935" if fps < 8 else ("#fdd835" if fps < 18 else "#43a047")
        self.fps_bar.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {color}, stop:1 #222); "
            f"border-radius:4px;"
        )
        self.fps_bar.setFixedWidth(max(6, w))

    def on_hazmat(self, texts):
        if texts:
            self.hazmat_box.setStyleSheet("background:#063; color:#adf7b6; border-radius:6px;")
            self.hazmat_box.setText("\n".join(texts))
        else:
            self.hazmat_box.setStyleSheet("background:#222; color:#0f0; border-radius:6px;")
            self.hazmat_box.setText("No hazmat detected")

    def on_qr(self, texts):
        if texts:
            self.qr_box.setStyleSheet("background:#664; color:#fff59d; border-radius:6px;")
            self.qr_box.setText("\n".join(texts))
        else:
            self.qr_box.setStyleSheet("background:#222; color:#ff0; border-radius:6px;")
            self.qr_box.setText("No QR codes detected")

    def _show_view_menu(self, pos: QPoint):
        menu = QMenu(self.view)
        act_snap = menu.addAction("Save Snapshot…")
        act_mirror = menu.addAction("Toggle Mirror")
        act_flip = menu.addAction("Toggle Flip")
        chosen = menu.exec(self.view.mapToGlobal(pos))
        if chosen is act_snap and self.view.pixmap():
            pm = self.view.pixmap()
            path, _ = QFileDialog.getSaveFileName(self, "Save Snapshot", "snapshot.jpg", "Images (*.png *.jpg)")
            if path: 
                pm.save(path)
        elif chosen is act_mirror:
            self.cb_hm.setChecked(not self.cb_hm.isChecked())
        elif chosen is act_flip:
            self.cb_vf.setChecked(not self.cb_vf.isChecked())

    def _current_profile(self) -> dict:
        return {
            "awb": int(self.cb_awb.isChecked()),
            "agc": int(self.cb_agc.isChecked()),
            "aec": int(self.cb_aec.isChecked()),
            "brightness": int(self.sl_bright.value()),
            "contrast":   int(self.sl_contr.value()),
            "saturation": int(self.sl_satur.value()),
            "ae_level":   int(self.sl_ae_level.value()),
            "aec_value":  int(self.sl_exp.value()),
            "agc_gain":   int(self.sl_gain.value()),
            "hmirror":    int(self.cb_hm.isChecked()),
            "vflip":      int(self.cb_vf.isChecked()),
        }

    def save_profile(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Profile", "profile.json", "JSON (*.json)")
        if not path:
            return
        
        worker = FileIOWorker('save', path, self._current_profile())
        worker.finished.connect(self._on_profile_saved)
        worker.start()
        self._file_worker = worker

    def _on_profile_saved(self, success: bool, msg: str):
        if success:
            print(f"Profile saved to {msg}")
        else:
            QMessageBox.warning(self, "Save Error", f"Failed to save profile: {msg}")
        if self._file_worker:
            self._file_worker.deleteLater()
            self._file_worker = None

    def load_profile(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Profile", "", "JSON (*.json)")
        if not path:
            return
        
        worker = FileIOWorker('load', path)
        worker.finished.connect(lambda success, msg: self._on_profile_loaded(success, msg, worker))
        worker.start()
        self._file_worker = worker

    def _on_profile_loaded(self, success: bool, msg: str, worker: FileIOWorker):
        if not success:
            QMessageBox.warning(self, "Load Error", f"Failed to load profile: {msg}")
            if self._file_worker:
                self._file_worker.deleteLater()
                self._file_worker = None

            return
        
        prof = worker.result
        self._apply_preset(
            aec=prof.get("aec",1), agc=prof.get("agc",1), awb=prof.get("awb",1),
            brightness=prof.get("brightness",0),
            contrast=prof.get("contrast",0),
            saturation=prof.get("saturation",0),
            ae_level=prof.get("ae_level",0),
            aec_value=prof.get("aec_value",300),
            agc_gain=prof.get("agc_gain",8),
        )
        self.cb_hm.setChecked(bool(prof.get("hmirror",0)))
        self.cb_vf.setChecked(bool(prof.get("vflip",0)))
        self._set_param(hmirror=int(self.cb_hm.isChecked()), vflip=int(self.cb_vf.isChecked()))
        if self._file_worker:
            self._file_worker.deleteLater()
            self._file_worker = None

class _Sparkline(QWidget):
    """Tiny sparkline widget for one channel."""
    def __init__(self, color="#80cbc4", max_samples=200, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(40)
        self.setMinimumWidth(200)
        self._data = deque(maxlen=max_samples)
        self._color = QColor(color)

    def append(self, v: float):
        self._data.append(v)
        self.update()

    def clear(self):
        self._data.clear()
        self.update()

    def paintEvent(self, event):
        if not self._data:
            return
        
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        vmin = min(self._data)
        vmax = max(self._data)
        rng = (vmax - vmin) or 1.0

        pen = QPen(self._color, 2.0)
        p.setPen(pen)

        def map_point(i, v, n):
            x = int(i * (w - 4) / max(1, n - 1)) + 2
            y = int(h - 2 - (v - vmin) * (h - 4) / rng)
            return x, y

        n = len(self._data)
        last_xy = None
        for i, v in enumerate(self._data):
            xy = map_point(i, v, n)
            if last_xy:
                p.drawLine(last_xy[0], last_xy[1], xy[0], xy[1])

            last_xy = xy

        p.end()


class MagnetometerPanel(QGroupBox):
    def __init__(self, title: str, url: str):
        super().__init__(title)
        self.worker: MagnetometerWorker | None = None

        self.url_edit = QLineEdit(url)
        self.url_edit.setStyleSheet(
            "font-size: 11px; font-family: monospace; color:#9efc6a; "
            "background:#1b1b1d; border:1px solid #555; border-radius:6px;"
        )
        self.btn_start = QPushButton("Connect")
        self.btn_stop  = QPushButton("Disconnect"); self.btn_stop.setEnabled(False)
        self.status    = QLabel("● disconnected"); self.status.setStyleSheet("color:#e53935; font-weight:bold;")
        self.hz_label  = QLabel("~ Hz: —")

        self.lab_x = QLabel("X: —")
        self.lab_y = QLabel("Y: —")
        self.lab_z = QLabel("Z: —")
        self.lab_x.setStyleSheet("color:#80cbc4; font: 600 14px 'Inter'")
        self.lab_y.setStyleSheet("color:#ffab91; font: 600 14px 'Inter'")
        self.lab_z.setStyleSheet("color:#fff59d; font: 600 14px 'Inter'")

        self.sp_x = _Sparkline("#80cbc4")
        self.sp_y = _Sparkline("#ffab91")
        self.sp_z = _Sparkline("#fff59d")

        top = QHBoxLayout()
        top.addWidget(QLabel("URL:"))
        top.addWidget(self.url_edit, 1)
        top.addWidget(self.btn_start)
        top.addWidget(self.btn_stop)

        vals = QHBoxLayout()
        vals.addWidget(self.lab_x)
        vals.addWidget(self.lab_y)
        vals.addWidget(self.lab_z)
        vals.addStretch(1)
        vals.addWidget(self.hz_label)
        vals.addWidget(self.status)

        root = QVBoxLayout()
        root.addLayout(top)
        root.addLayout(vals)
        root.addWidget(self.sp_x)
        root.addWidget(self.sp_y)
        root.addWidget(self.sp_z)
        self.setLayout(root)

        self.btn_start.clicked.connect(self.start_stream)
        self.btn_stop .clicked.connect(self.stop_stream)

    def start_stream(self):
        if self.worker and self.worker.isRunning():
            return
        
        url = self.url_edit.text().strip()
        if not url:
            return
        
        self.worker = MagnetometerWorker(url)
        self.worker.reading.connect(self.on_reading, Qt.ConnectionType.QueuedConnection)
        self.worker.rate.connect(self.on_rate, Qt.ConnectionType.QueuedConnection)
        self.worker.status.connect(self.on_status, Qt.ConnectionType.QueuedConnection)
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_status("● connecting…", "#fdd835")

    def stop_stream(self):
        if self.worker:
            self.worker.stop()
            self.worker = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_status("● disconnected", "#e53935")
        self.hz_label.setText("~ Hz: —")
        self.sp_x.clear()
        self.sp_y.clear()
        self.sp_z.clear()

    def on_status(self, ok: bool, msg: str):
        color = "#43a047" if ok else "#fdd835"
        self._set_status(f"● {msg}", color)

    def on_rate(self, hz: float):
        self.hz_label.setText(f"~ Hz: {hz:.1f}")

    def on_reading(self, x: float, y: float, z: float):
        self.lab_x.setText(f"X: {x:6.2f} µT")
        self.lab_y.setText(f"Y: {y:6.2f} µT")
        self.lab_z.setText(f"Z: {z:6.2f} µT")
        self.sp_x.append(x)
        self.sp_y.append(y)
        self.sp_z.append(z)
        self._set_status("● connected", "#43a047")

    def _set_status(self, txt: str, color_hex: str):
        self.status.setText(txt)
        self.status.setStyleSheet(f"color:{color_hex}; font-weight:bold;")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESP32-CAM Multi-Camera (Live)")
        self.setMinimumSize(QSize(1600, 900))

        grid = QGridLayout()
        grid.addWidget(CameraPanel("Camera 1", "http://192.168.50.231:81/stream"), 0, 0)
        grid.addWidget(CameraPanel("Camera 2", "http://192.168.50.232:81/stream"), 0, 1)
        grid.addWidget(MagnetometerPanel("FXOS8700 Magnetometer", "http://192.168.50.143/magnetometer/stream"), 1, 0, 1, 2)

        container = QWidget()
        container.setLayout(grid)
        self.setCentralWidget(container)

        tb = QToolBar("Controls")
        self.addToolBar(tb)
        act_start_all = QAction("Start All", self)
        act_stop_all = QAction("Stop All", self)
        act_snap = QAction("Snapshot", self)
        act_full = QAction("Fullscreen", self)
        tb.addAction(act_start_all)
        tb.addAction(act_stop_all)
        tb.addSeparator()
        tb.addAction(act_snap)
        tb.addSeparator()
        tb.addAction(act_full)

        self.panels = []
        central_grid = self.centralWidget().layout()
        for r in range(central_grid.rowCount()):
            for c in range(central_grid.columnCount()):
                item = central_grid.itemAtPosition(r, c)
                if item and item.widget() and isinstance(item.widget(), CameraPanel):
                    self.panels.append(item.widget())

        def start_all():
            for p in self.panels:
                p.start_stream()

        def stop_all():
            for p in self.panels:
                p.stop_stream()

        def toggle_full():
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

        def snapshot_active():
            for p in self.panels:
                if p.worker and p.worker.isRunning() and p.view.pixmap():
                    pm = p.view.pixmap()
                    path, _ = QFileDialog.getSaveFileName(self, "Save Snapshot", "snapshot.jpg", "Images (*.png *.jpg)")
                    if path:
                        pm.save(path)

                    break

        act_start_all.triggered.connect(start_all)
        act_stop_all.triggered.connect(stop_all)
        act_full.triggered.connect(toggle_full)
        act_snap.triggered.connect(snapshot_active)

        QShortcut(QKeySequence("F"), self, activated=toggle_full)
        QShortcut(QKeySequence("S"), self, activated=snapshot_active)
        
        def space_toggle():
            if not self.panels:
                return
            
            p = self.panels[0]
            if p.worker and p.worker.isRunning():
                p.stop_stream()
            else:
                p.start_stream()

        QShortcut(QKeySequence("Space"), self, activated=space_toggle)

    def closeEvent(self, event):
        for p in getattr(self, "panels", []):
            try:
                if hasattr(p, "worker") and p.worker:
                    p.worker.stop()

                if getattr(p, "_param_setter", None):
                    ps = p._param_setter
                    if ps.isRunning():
                        try:
                            ps.requestInterruption()
                        except Exception:
                            pass

                        ps.wait(300)
                        if ps.isRunning():
                            try:
                                ps.terminate()
                            except Exception:
                                pass

                            ps.wait(200)

            except Exception as e:
                print("[closeEvent-stop]", e)

        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QMainWindow { background-color: #121212; }
        QGroupBox {
            border: 2px solid #333; border-radius: 10px;
            margin-top: 10px; background-color: #1d1d1f; color: #ddd;
            font-weight: 600; font-size: 14px;
        }
        QLabel { color: #ccc; font-size: 14px; }
        QPushButton {
            background-color: #2b2b2e; color: #eee; border: 1px solid #555;
            border-radius: 6px; padding: 6px 12px;
        }
        QPushButton:hover { background-color: #3a3a3d; }
        QLineEdit {
            background-color: #1b1b1d; color: #9efc6a; border: 1px solid #555;
            border-radius: 6px; padding: 4px 6px; font-family: monospace;
            font-size: 11px;
        }
        QTextEdit {
            border-radius: 6px; font-family: monospace; font-size: 12px;
        }
    """)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()