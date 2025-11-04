from __future__ import annotations
import time
from typing import Optional, Tuple, List, Callable
import cv2
import numpy as np
import requests
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

def extract_host_from_stream_url(url: str) -> str:
    if not url:
        return ""
    hostpart = url.split("://", 1)[-1]
    hostpart = hostpart.split("/", 1)[0]
    host = hostpart.split(":", 1)[0]
    return host

class ParameterSetterThread(QThread):
    finished = pyqtSignal(bool)

    def __init__(self, host: str, params: dict, timeout: float = 3.0):
        super().__init__()
        self.host = host
        self.params = params or {}
        self.timeout = timeout
        self._interrupted = False

    def requestInterruption(self):
        self._interrupted = True

    def run(self):
        if not self.host:
            self.finished.emit(False)
            return

        url = f"http://{self.host}/set"
        try:
            if self._interrupted:
                self.finished.emit(False)
                return

            resp = requests.get(url, params=self.params, timeout=(1.5, self.timeout))
            if self._interrupted:
                self.finished.emit(False)
                return

            ok = resp.status_code == 200
            self.finished.emit(ok)
        except requests.RequestException as e:
            print("[esp32cam.set_params] request failed:", e)
            self.finished.emit(False)
        except Exception as e:
            print("[esp32cam.set_params] unexpected:", e)
            self.finished.emit(False)
        finally:
            return

class ESP32CamClient:
    def __init__(self, stream_url: str, timeout: float = 3.0):
        self.stream_url = stream_url
        self.timeout = timeout
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.stream_url)
        return bool(self._cap and self._cap.isOpened())

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._cap:
            return False, None
        ok, frame = self._cap.read()
        if not ok:
            return False, None
        return True, frame

    def close(self):
        if self._cap:
            self._cap.release()
            self._cap = None

class CameraWorker(QThread):
    frameReady = pyqtSignal(np.ndarray)
    fpsReady   = pyqtSignal(float)
    opened     = pyqtSignal(bool, str)
    hazmatDet  = pyqtSignal(list)
    qrDet      = pyqtSignal(list)

    def __init__(self, stream_url: str, hazmat_factory: Optional[Callable[[], object]] = None, qr_factory: Optional[Callable[[], object]] = None, target_fps: float = 30.0):
        super().__init__()
        self.client = ESP32CamClient(stream_url)
        self._running = False
        self._mutex = QMutex()
        
        self.target_fps = target_fps
        self.min_frame_interval = 1.0 / target_fps

        self.hazmat_factory = hazmat_factory
        self.qr_factory = qr_factory

        self._enable_hazmat = False
        self._enable_qr = False
        self.hazmat = None
        self.qr = None

    @property
    def enable_hazmat(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._enable_hazmat
    
    @enable_hazmat.setter
    def enable_hazmat(self, value: bool):
        with QMutexLocker(self._mutex):
            self._enable_hazmat = value

    @property
    def enable_qr(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._enable_qr
    
    @enable_qr.setter
    def enable_qr(self, value: bool):
        with QMutexLocker(self._mutex):
            self._enable_qr = value

    def run(self):
        self._running = True
        if not self.client.open():
            self.opened.emit(False, f"Could not open stream {self.client.stream_url}")
            return

        self.opened.emit(True, "opened")
        last_frame_time = time.time()
        last_fps_time = time.time()
        frame_count = 0
        fps_avg = 0.0

        while self._running:
            current_time = time.time()
            
            if current_time - last_frame_time < self.min_frame_interval:
                self.msleep(5)
                continue

            ok, frame = self.client.read()
            if not ok or frame is None:
                self.msleep(10)
                continue

            frame_count += 1
            last_frame_time = current_time

            haz_texts: List[str] = []
            qr_texts: List[str] = []

            enable_haz = self.enable_hazmat
            enable_qr = self.enable_qr

            if enable_haz and self.hazmat is None and self.hazmat_factory:
                try:
                    self.hazmat = self.hazmat_factory()
                except Exception as e:
                    print("[hazmat] init error:", e)
                    self.enable_hazmat = False
                    enable_haz = False

            if enable_qr and self.qr is None and self.qr_factory:
                try:
                    self.qr = self.qr_factory()
                except Exception as e:
                    print("[qr] init error:", e)
                    self.enable_qr = False
                    enable_qr = False

            if enable_haz and self.hazmat:
                try:
                    cids, confs, boxes = self.hazmat.detect(frame)
                    frame = self.hazmat.annotate(frame, cids, confs, boxes)
                    if hasattr(self.hazmat, "labels"):
                        haz_texts = [
                            f"{self.hazmat.labels[cid]} ({conf:.2f})"
                            for cid, conf in zip(cids, confs)
                            if 0 <= cid < len(self.hazmat.labels)
                        ]
                    else:
                        haz_texts = [f"id{int(cid)} ({confs[i]:.2f})" for i, cid in enumerate(cids)]
                except Exception as e:
                    print("[hazmat] run error:", e)

            if enable_qr and self.qr:
                try:
                    qr_res = self.qr.detect(frame)
                    frame = self.qr.annotate(frame, qr_res)
                    qr_texts = [txt for txt, _ in qr_res if txt]
                except Exception as e:
                    print("[qr] run error:", e)

            if current_time - last_fps_time >= 1.0:
                fps_avg = frame_count / (current_time - last_fps_time)
                self.fpsReady.emit(fps_avg)
                frame_count = 0
                last_fps_time = current_time

            self.hazmatDet.emit(haz_texts)
            self.qrDet.emit(qr_texts)
            self.frameReady.emit(frame)

        self.client.close()

    def stop(self):
        self._running = False
        self.wait(1000)