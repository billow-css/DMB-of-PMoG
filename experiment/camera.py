#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment/camera.py
====================
相机通信抽象层。

- ``StubCamera``：无真机联调
- ``EdsdkCamera``：合并本地 EDSDK 13.20.21（ctypes），支持 R7 有线遥控 AF / 快门 / 回传
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from abc import ABC, abstractmethod
from ctypes import byref, c_int, c_ubyte, c_uint, c_ulonglong, c_void_p, sizeof
from pathlib import Path
from typing import Optional

log = logging.getLogger("experiment.camera")


class CameraBase(ABC):
    """USB 遥控相机接口。"""

    @abstractmethod
    def connect(self) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def autofocus(self) -> None:
        """半按快门 / Evf AF。"""

    @abstractmethod
    def shoot(self, dest: Path) -> Path:
        """全按快门并将文件保存到 dest。"""

    @property
    def connected(self) -> bool:
        return False

    @property
    def liveview_on(self) -> bool:
        return False

    def start_liveview(self) -> None:
        """开启 PC Live View（可选）。"""

    def stop_liveview(self) -> None:
        """关闭 PC Live View（可选）。"""

    def grab_evf_jpeg(self, retries: int = 8) -> Optional[bytes]:
        """抓一帧 EVF JPEG；未就绪时返回 None。"""
        return None

    def __enter__(self) -> "CameraBase":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class StubCamera(CameraBase):
    """无真机时的占位实现。"""

    def __init__(
        self,
        af_delay_s: float = 0.15,
        shoot_delay_s: float = 0.35,
        write_placeholder: bool = True,
    ):
        self.af_delay_s = af_delay_s
        self.shoot_delay_s = shoot_delay_s
        self.write_placeholder = write_placeholder
        self._connected = False
        self._evf_on = False
        self.shots = 0

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def liveview_on(self) -> bool:
        return self._evf_on

    def connect(self) -> None:
        log.info("[stub] camera connect (no USB device)")
        self._connected = True

    def close(self) -> None:
        if self._connected:
            log.info("[stub] camera close | total_shots=%d", self.shots)
        self._evf_on = False
        self._connected = False

    def start_liveview(self) -> None:
        self._ensure()
        self._evf_on = True
        log.info("[stub] live view on")

    def stop_liveview(self) -> None:
        self._evf_on = False
        log.info("[stub] live view off")

    def grab_evf_jpeg(self, retries: int = 8) -> Optional[bytes]:
        if not self._connected or not self._evf_on:
            return None
        try:
            from io import BytesIO

            from PIL import Image, ImageDraw

            img = Image.new("RGB", (640, 420), (28, 36, 48))
            draw = ImageDraw.Draw(img)
            t = time.strftime("%H:%M:%S")
            draw.rectangle([20, 20, 620, 400], outline=(80, 160, 220), width=2)
            draw.text((40, 40), f"StubCamera EVF  {t}", fill=(200, 220, 240))
            draw.text((40, 80), f"shots={self.shots}", fill=(160, 180, 200))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return buf.getvalue()
        except Exception:
            return bytes([0xFF, 0xD8, 0xFF, 0xD9])

    def autofocus(self) -> None:
        self._ensure()
        log.info("[stub] AF …")
        time.sleep(self.af_delay_s)
        log.info("[stub] AF done")

    def shoot(self, dest: Path) -> Path:
        self._ensure()
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("[stub] SHOOT → %s", dest)
        time.sleep(self.shoot_delay_s)
        if self.write_placeholder:
            dest.write_bytes(bytes([0xFF, 0xD8, 0xFF, 0xD9]))
        else:
            dest.write_bytes(b"")
        self.shots += 1
        log.info("[stub] SHOOT done | #%d", self.shots)
        return dest

    def _ensure(self) -> None:
        if not self._connected:
            raise RuntimeError("camera not connected")


class EdsdkCamera(CameraBase):
    """
    Canon EDSDK 遥控（Windows）。

    屏摄默认使用 ``Completely_NonAF``：屏幕距离固定，避免 AF_NG(0x8D01) 打断连拍。
    """

    def __init__(
        self,
        library_path: Optional[str] = None,
        download_timeout_s: float = 30.0,
        keep_liveview: bool = False,
        shutter_non_af: bool = True,
        shoot_retries: int = 3,
        post_download_pump_s: float = 0.05,
    ):
        self.library_path = library_path  # Dll 目录或 EDSDK.dll 路径
        self.download_timeout_s = float(download_timeout_s)
        self.keep_liveview = bool(keep_liveview)
        self.shutter_non_af = bool(shutter_non_af)
        self.shoot_retries = max(1, int(shoot_retries))
        self.post_download_pump_s = max(0.0, float(post_download_pump_s))

        self._lib = None
        self._camera = None
        self._camera_list = None
        self._connected = False
        self._evf_on = False
        self._lock = threading.RLock()
        self._object_cb = None  # 保持回调引用防 GC
        self._property_cb = None
        self._evf_device_ready = threading.Event()
        self._pending_dest: Optional[Path] = None
        self._download_done = threading.Event()
        self._download_error: Optional[str] = None
        self._last_saved: Optional[Path] = None
        self._last_evf_error: Optional[str] = None
        self.shots = 0
        self.model_name = ""

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def liveview_on(self) -> bool:
        return self._evf_on

    def connect(self) -> None:
        import edsdk_raw as eds

        with self._lock:
            if self._connected:
                return

            dll_dir = None
            if self.library_path:
                p = Path(self.library_path)
                dll_dir = p if p.is_dir() else p.parent

            # COM（官方 Sample 要求）
            try:
                import ctypes

                ctypes.windll.ole32.CoInitializeEx(None, 0x0)  # COINIT_MULTITHREADED=0
            except Exception as e:
                log.warning("CoInitializeEx: %s", e)

            self._lib = eds.load_edsdk(dll_dir)
            eds.check(self._lib.EdsInitializeSDK(), "EdsInitializeSDK")

            cam_list = eds.EdsCameraListRef()
            eds.check(self._lib.EdsGetCameraList(byref(cam_list)), "EdsGetCameraList")
            self._camera_list = cam_list

            count = c_int(0)
            eds.check(
                self._lib.EdsGetChildCount(cam_list, byref(count)), "EdsGetChildCount"
            )
            if count.value < 1:
                self._lib.EdsRelease(cam_list)
                self._lib.EdsTerminateSDK()
                raise RuntimeError(
                    "未检测到相机：请用数据线连接 EOS R7，关闭 EOS Utility，"
                    "相机电源打开，USB 设为可遥控。"
                )

            camera = eds.EdsCameraRef()
            eds.check(
                self._lib.EdsGetChildAtIndex(cam_list, 0, byref(camera)),
                "EdsGetChildAtIndex",
            )
            self._camera = camera

            info = eds.EdsDeviceInfo()
            eds.check(
                self._lib.EdsGetDeviceInfo(camera, byref(info)), "EdsGetDeviceInfo"
            )
            self.model_name = info.szDeviceDescription.decode("utf-8", errors="replace")
            log.info("[edsdk] 检测到: %s", self.model_name)

            # 对象事件：Host 回传
            self._object_cb = eds.OBJECT_EVENT_HANDLER(self._on_object_event)
            eds.check(
                self._lib.EdsSetObjectEventHandler(
                    camera,
                    eds.kEdsObjectEvent_All,
                    self._object_cb,
                    None,
                ),
                "EdsSetObjectEventHandler",
            )
            # 属性事件：等待 Evf_OutputDevice 生效（否则 DownloadEvf 会一直 NOTREADY）
            self._property_cb = eds.PROPERTY_EVENT_HANDLER(self._on_property_event)
            eds.check(
                self._lib.EdsSetPropertyEventHandler(
                    camera,
                    eds.kEdsPropertyEvent_All,
                    self._property_cb,
                    None,
                ),
                "EdsSetPropertyEventHandler",
            )

            eds.retry_busy(
                lambda: self._lib.EdsOpenSession(camera), "EdsOpenSession"
            )

            # SaveTo Host + 容量（官方 OpenSessionCommand）
            save_to = c_uint(eds.kEdsSaveTo_Host)
            eds.retry_busy(
                lambda: self._lib.EdsSetPropertyData(
                    camera,
                    eds.kEdsPropID_SaveTo,
                    0,
                    sizeof(save_to),
                    byref(save_to),
                ),
                "Set SaveTo=Host",
            )
            cap = eds.EdsCapacity(0x7FFFFFFF, 0x1000, 1)
            eds.check(self._lib.EdsSetCapacity(camera, cap), "EdsSetCapacity")

            if self.keep_liveview:
                self._start_liveview()
            else:
                log.info("[edsdk] 跳过 PC LiveView（加快 Host 回传）")
            self._connected = True
            log.info("[edsdk] session open | liveview=%s", self._evf_on)

    def close(self) -> None:
        import edsdk_raw as eds

        with self._lock:
            if not self._connected and self._lib is None:
                return
            try:
                if self._camera and self._evf_on:
                    self._stop_liveview()
                if self._camera:
                    self._lib.EdsCloseSession(self._camera)
                    self._lib.EdsRelease(self._camera)
                    self._camera = None
                if self._camera_list:
                    self._lib.EdsRelease(self._camera_list)
                    self._camera_list = None
                if self._lib:
                    self._lib.EdsTerminateSDK()
                    self._lib = None
            except Exception:
                log.exception("[edsdk] close error")
            self._connected = False
            self._object_cb = None
            self._property_cb = None
            log.info("[edsdk] closed | total_shots=%d", self.shots)

    def autofocus(self) -> None:
        import edsdk_raw as eds

        with self._lock:
            self._ensure()
            cam = self._camera
            # 延长自动关机计时，大批量连拍时防休眠
            self._lib.EdsSendCommand(
                cam, eds.kEdsCameraCommand_ExtendShutDownTimer, 0
            )
            if self.shutter_non_af:
                # 电子快门屏摄：不对焦、不半按；仅延长自动关机（由 server 偶发调用）
                log.info("[edsdk] AF skip (shutter_non_af=True，仅 ExtendShutDownTimer)")
                return

            log.info("[edsdk] AF …")
            err = self._lib.EdsSendCommand(
                cam, eds.kEdsCameraCommand_DoEvfAf, eds.kEdsCameraCommand_EvfAf_ON
            )
            self._pump(0.25)
            self._lib.EdsSendCommand(
                cam, eds.kEdsCameraCommand_DoEvfAf, eds.kEdsCameraCommand_EvfAf_OFF
            )
            if err != eds.EDS_ERR_OK:
                log.warning("DoEvfAf=0x%08X，改用半按快门", err)
                self._lib.EdsSendCommand(
                    cam,
                    eds.kEdsCameraCommand_PressShutterButton,
                    eds.kEdsCameraCommand_ShutterButton_Halfway,
                )
                self._pump(0.25)
                self._lib.EdsSendCommand(
                    cam,
                    eds.kEdsCameraCommand_PressShutterButton,
                    eds.kEdsCameraCommand_ShutterButton_OFF,
                )
            self._pump(0.05)
            log.info("[edsdk] AF done")

    def shoot(self, dest: Path) -> Path:
        import edsdk_raw as eds

        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        last_err: Optional[Exception] = None

        for attempt in range(1, self.shoot_retries + 1):
            try:
                return self._shoot_once(dest, attempt=attempt)
            except Exception as e:
                last_err = e
                log.warning(
                    "[edsdk] SHOOT 第 %d/%d 次失败: %s",
                    attempt,
                    self.shoot_retries,
                    e,
                )
                with self._lock:
                    self._release_shutter()
                    self._pump(0.4)
                if attempt < self.shoot_retries:
                    time.sleep(0.5 * attempt)
        assert last_err is not None
        raise last_err

    def _release_shutter(self) -> None:
        import edsdk_raw as eds

        if self._lib and self._camera:
            try:
                self._lib.EdsSendCommand(
                    self._camera,
                    eds.kEdsCameraCommand_PressShutterButton,
                    eds.kEdsCameraCommand_ShutterButton_OFF,
                )
            except Exception:
                pass

    def _shoot_once(self, dest: Path, attempt: int = 1) -> Path:
        import edsdk_raw as eds

        with self._lock:
            self._ensure()
            self._pending_dest = dest
            self._download_error = None
            self._download_done.clear()
            self._last_saved = None

            cam = self._camera
            mode = (
                eds.kEdsCameraCommand_ShutterButton_Completely_NonAF
                if self.shutter_non_af
                else eds.kEdsCameraCommand_ShutterButton_Completely
            )
            mode_name = "Completely_NonAF" if self.shutter_non_af else "Completely"
            log.info("[edsdk] SHOOT (%s) try=%d → %s", mode_name, attempt, dest)

            # 不必每张 ExtendShutDownTimer（省一次往返）
            if self.shots % 25 == 0:
                self._lib.EdsSendCommand(
                    cam, eds.kEdsCameraCommand_ExtendShutDownTimer, 0
                )

            err = self._lib.EdsSendCommand(
                cam, eds.kEdsCameraCommand_PressShutterButton, mode
            )
            # AF_NG：强制改 NonAF 再试一次（本 attempt 内）
            if err == eds.EDS_ERR_TAKE_PICTURE_AF_NG:
                log.warning("[edsdk] AF_NG(0x8D01)，改用 Completely_NonAF 重按")
                self._release_shutter()
                self._pump(0.2)
                err = self._lib.EdsSendCommand(
                    cam,
                    eds.kEdsCameraCommand_PressShutterButton,
                    eds.kEdsCameraCommand_ShutterButton_Completely_NonAF,
                )
            if err == eds.EDS_ERR_DEVICE_BUSY:
                self._release_shutter()
                self._pump(0.5)
                err = self._lib.EdsSendCommand(
                    cam,
                    eds.kEdsCameraCommand_PressShutterButton,
                    eds.kEdsCameraCommand_ShutterButton_Completely_NonAF,
                )
            self._release_shutter()
            if err != eds.EDS_ERR_OK:
                raise RuntimeError(f"EDSDK Shutter failed: 0x{err:08X}")

            # 等待 ObjectEvent 下载（电子快门：更密轮询，尽早发现落盘）
            deadline = time.time() + self.download_timeout_s
            while time.time() < deadline:
                self._lib.EdsGetEvent()
                if self._download_done.wait(0.012):
                    break
            else:
                raise TimeoutError(
                    f"等待相机回传超时 ({self.download_timeout_s}s)：{dest}"
                )

            if self._download_error:
                raise RuntimeError(self._download_error)

            # 下载后再泵一会，让机身离开 BUSY（电子快门可很短）
            if self.post_download_pump_s > 0:
                self._pump(self.post_download_pump_s)

            saved = self._last_saved or dest
            if saved.resolve() != dest.resolve():
                if saved.is_file():
                    if dest.suffix.lower() != saved.suffix.lower():
                        final = dest.with_suffix(saved.suffix)
                        if final.resolve() != saved.resolve():
                            shutil.move(str(saved), str(final))
                            saved = final
                        if dest.suffix.lower() in {".jpg", ".jpeg"} and saved.suffix.lower() not in {
                            ".jpg",
                            ".jpeg",
                        }:
                            log.warning(
                                "相机输出 %s（非 JPEG）。已保存为 %s；"
                                "如需 JPEG 请在机身画质设为 JPEG/Fine。",
                                saved.suffix,
                                saved,
                            )
                            self.shots += 1
                            return saved
                    shutil.move(str(saved), str(dest))
                    saved = dest

            if not Path(saved).is_file():
                raise RuntimeError(f"回传文件不存在: {saved}")
            self.shots += 1
            log.info("[edsdk] SHOOT done | #%d → %s", self.shots, saved)
            return Path(saved)

    # ------------------------------------------------------------------
    def _ensure(self) -> None:
        if not self._connected or self._camera is None or self._lib is None:
            raise RuntimeError("camera not connected")

    def _pump(self, seconds: float) -> None:
        end = time.time() + max(0.0, seconds)
        while time.time() < end:
            if self._lib:
                self._lib.EdsGetEvent()
            time.sleep(0.02)

    def start_liveview(self) -> None:
        with self._lock:
            self._ensure()
            self._start_liveview()

    def stop_liveview(self) -> None:
        with self._lock:
            if self._connected and self._evf_on:
                self._stop_liveview()

    def grab_evf_jpeg(self, retries: int = 30) -> Optional[bytes]:
        """从相机下载一帧 Live View JPEG。"""
        import edsdk_raw as eds

        with self._lock:
            self._ensure()
            if not self._evf_on:
                self._start_liveview()
                if not self._evf_on:
                    self._last_evf_error = "liveview start failed"
                    return None

            lib = self._lib
            cam = self._camera
            assert lib is not None and cam is not None

            # 确认 PC / PC_Small 输出仍在（机身偶发会清掉）
            device = c_uint(0)
            lib.EdsGetPropertyData(
                cam, eds.kEdsPropID_Evf_OutputDevice, 0, sizeof(device), byref(device)
            )
            pc_bits = eds.kEdsEvfOutputDevice_PC | eds.kEdsEvfOutputDevice_PC_Small
            if (device.value & pc_bits) == 0:
                log.info("[edsdk] Evf_OutputDevice 无 PC 输出，重新开启 (cur=0x%X)", device.value)
                self._evf_on = False
                self._start_liveview()
                if not self._evf_on:
                    self._last_evf_error = "Evf_OutputDevice PC bit missing"
                    return None

            stream = eds.EdsStreamRef()
            evf = eds.EdsEvfImageRef()
            err = lib.EdsCreateMemoryStream(
                c_ulonglong(2 * 1024 * 1024), byref(stream)
            )
            if err != eds.EDS_ERR_OK:
                self._last_evf_error = f"CreateMemoryStream 0x{err:08X}"
                return None
            err = lib.EdsCreateEvfImageRef(stream, byref(evf))
            if err != eds.EDS_ERR_OK:
                lib.EdsRelease(stream)
                self._last_evf_error = f"CreateEvfImageRef 0x{err:08X}"
                return None

            last_err = 0
            data: Optional[bytes] = None
            try:
                for attempt in range(max(1, retries)):
                    lib.EdsGetEvent()
                    err = lib.EdsDownloadEvfImage(cam, evf)
                    last_err = err
                    if err == eds.EDS_ERR_OK:
                        # 下载后指针可能在末尾，先 Seek 到头再读
                        try:
                            lib.EdsSeek(stream, 0, eds.kEdsSeek_Begin)
                        except Exception:
                            pass
                        ptr = c_void_p()
                        length = c_ulonglong(0)
                        e1 = lib.EdsGetPointer(stream, byref(ptr))
                        e2 = lib.EdsGetLength(stream, byref(length))
                        if (
                            e1 == eds.EDS_ERR_OK
                            and e2 == eds.EDS_ERR_OK
                            and ptr.value
                            and length.value
                        ):
                            n = int(length.value)
                            buf = (c_ubyte * n).from_address(ptr.value)
                            raw = bytes(buf)
                            data = self._trim_jpeg(raw)
                            if data and data[:2] == b"\xff\xd8":
                                self._last_evf_error = None
                                return data
                            self._last_evf_error = (
                                f"bad jpeg len={n} head={raw[:8]!r}"
                            )
                        else:
                            self._last_evf_error = (
                                f"GetPointer/Length 0x{e1:08X}/0x{e2:08X} len={length.value}"
                            )
                        break
                    if err in (
                        eds.EDS_ERR_OBJECT_NOTREADY,
                        eds.EDS_ERR_DEVICE_BUSY,
                    ):
                        time.sleep(0.05)
                        continue
                    self._last_evf_error = f"DownloadEvfImage 0x{err:08X}"
                    log.warning("[edsdk] DownloadEvfImage 0x%08X", err)
                    break
                else:
                    self._last_evf_error = f"NOTREADY x{retries} last=0x{last_err:08X}"
            finally:
                if evf:
                    lib.EdsRelease(evf)
                if stream:
                    lib.EdsRelease(stream)
            return None

    @staticmethod
    def _trim_jpeg(data: bytes) -> bytes:
        start = data.find(b"\xff\xd8")
        if start < 0:
            return data
        end = data.rfind(b"\xff\xd9")
        if end > start:
            return data[start : end + 2]
        return data[start:]

    def _start_liveview(self) -> None:
        import edsdk_raw as eds

        if self._evf_on:
            return
        cam = self._camera
        lib = self._lib
        assert cam is not None and lib is not None

        self._evf_device_ready.clear()

        mode = c_uint(0)
        lib.EdsGetPropertyData(
            cam, eds.kEdsPropID_Evf_Mode, 0, sizeof(mode), byref(mode)
        )
        if mode.value == 0:
            mode = c_uint(1)
            err = lib.EdsSetPropertyData(
                cam, eds.kEdsPropID_Evf_Mode, 0, sizeof(mode), byref(mode)
            )
            if err != eds.EDS_ERR_OK:
                log.warning("Evf_Mode set 0x%08X（继续尝试）", err)
            else:
                # 等属性回调 / 事件泵
                self._pump(0.2)

        device = c_uint(0)
        lib.EdsGetPropertyData(
            cam, eds.kEdsPropID_Evf_OutputDevice, 0, sizeof(device), byref(device)
        )
        if (device.value & eds.kEdsEvfOutputDevice_PC) == 0:
            device = c_uint(device.value | eds.kEdsEvfOutputDevice_PC)
            err = lib.EdsSetPropertyData(
                cam, eds.kEdsPropID_Evf_OutputDevice, 0, sizeof(device), byref(device)
            )
            if err != eds.EDS_ERR_OK:
                log.warning("Evf_OutputDevice PC 0x%08X", err)
                self._evf_on = False
                self._last_evf_error = f"Evf_OutputDevice set 0x{err:08X}"
                return

        # 等到属性变更或读回确认 PC 位（R7 常需 0.5–1.5s）
        deadline = time.time() + 2.5
        ready = False
        while time.time() < deadline:
            lib.EdsGetEvent()
            if self._evf_device_ready.is_set():
                ready = True
                break
            cur = c_uint(0)
            lib.EdsGetPropertyData(
                cam, eds.kEdsPropID_Evf_OutputDevice, 0, sizeof(cur), byref(cur)
            )
            if cur.value & eds.kEdsEvfOutputDevice_PC:
                ready = True
                break
            time.sleep(0.05)

        if not ready:
            log.warning("[edsdk] 等待 Evf_OutputDevice=PC 超时，仍尝试抓帧")

        # 再泵一会让首帧缓冲就绪
        self._pump(0.5)
        self._evf_on = True
        out_dev = c_uint(0)
        lib.EdsGetPropertyData(
            cam, eds.kEdsPropID_Evf_OutputDevice, 0, sizeof(out_dev), byref(out_dev)
        )
        mode_now = c_uint(0)
        lib.EdsGetPropertyData(
            cam, eds.kEdsPropID_Evf_Mode, 0, sizeof(mode_now), byref(mode_now)
        )
        log.info(
            "[edsdk] live view → PC (ready=%s, OutputDevice=0x%X, Evf_Mode=%d)",
            ready,
            out_dev.value,
            mode_now.value,
        )

        # 预热：丢掉 NOTREADY，确认能下到至少一帧
        warmed = False
        for _ in range(50):
            lib.EdsGetEvent()
            probe = self._download_evf_once_unlocked()
            if probe:
                warmed = True
                log.info("[edsdk] EVF 首帧 OK (%d bytes)", len(probe))
                break
            time.sleep(0.04)
        if not warmed:
            log.info("[edsdk] EVF 回退尝试 OutputDevice=PC-only")
            out = c_uint(eds.kEdsEvfOutputDevice_PC)
            lib.EdsSetPropertyData(
                cam, eds.kEdsPropID_Evf_OutputDevice, 0, sizeof(out), byref(out)
            )
            self._pump(0.6)
            for _ in range(40):
                lib.EdsGetEvent()
                probe = self._download_evf_once_unlocked()
                if probe:
                    warmed = True
                    log.info("[edsdk] EVF 首帧 OK via PC-only (%d bytes)", len(probe))
                    break
                time.sleep(0.04)
        if not warmed:
            log.info("[edsdk] EVF 回退尝试 OutputDevice=PC_Small")
            out = c_uint(eds.kEdsEvfOutputDevice_PC_Small)
            lib.EdsSetPropertyData(
                cam, eds.kEdsPropID_Evf_OutputDevice, 0, sizeof(out), byref(out)
            )
            self._pump(0.6)
            for _ in range(40):
                lib.EdsGetEvent()
                probe = self._download_evf_once_unlocked()
                if probe:
                    warmed = True
                    log.info("[edsdk] EVF 首帧 OK via PC_Small (%d bytes)", len(probe))
                    break
                time.sleep(0.04)
        if not warmed:
            log.warning(
                "[edsdk] EVF 预热未拿到帧: %s（界面仍会继续重试）",
                self._last_evf_error,
            )

    def _download_evf_once_unlocked(self) -> Optional[bytes]:
        """已持锁；单次尝试下载（用于预热）。"""
        import edsdk_raw as eds

        lib = self._lib
        cam = self._camera
        if not lib or not cam:
            return None
        stream = eds.EdsStreamRef()
        evf = eds.EdsEvfImageRef()
        if lib.EdsCreateMemoryStream(c_ulonglong(2 * 1024 * 1024), byref(stream)):
            return None
        if lib.EdsCreateEvfImageRef(stream, byref(evf)):
            lib.EdsRelease(stream)
            return None
        err = lib.EdsDownloadEvfImage(cam, evf)
        data = None
        if err == eds.EDS_ERR_OK:
            try:
                lib.EdsSeek(stream, 0, eds.kEdsSeek_Begin)
            except Exception:
                pass
            ptr = c_void_p()
            length = c_ulonglong(0)
            if (
                lib.EdsGetPointer(stream, byref(ptr)) == eds.EDS_ERR_OK
                and lib.EdsGetLength(stream, byref(length)) == eds.EDS_ERR_OK
                and ptr.value
                and length.value
            ):
                raw = bytes((c_ubyte * int(length.value)).from_address(ptr.value))
                data = self._trim_jpeg(raw)
                if not (data and data[:2] == b"\xff\xd8"):
                    data = None
            self._last_evf_error = None if data else "warmup bad jpeg"
        else:
            self._last_evf_error = f"warmup 0x{err:08X}"
        lib.EdsRelease(evf)
        lib.EdsRelease(stream)
        return data

    def _stop_liveview(self) -> None:
        import edsdk_raw as eds

        if not self._camera or not self._lib:
            return
        device = c_uint(0)
        self._lib.EdsGetPropertyData(
            self._camera,
            eds.kEdsPropID_Evf_OutputDevice,
            0,
            sizeof(device),
            byref(device),
        )
        device = c_uint(device.value & ~eds.kEdsEvfOutputDevice_PC)
        self._lib.EdsSetPropertyData(
            self._camera,
            eds.kEdsPropID_Evf_OutputDevice,
            0,
            sizeof(device),
            byref(device),
        )
        self._evf_on = False
        self._evf_device_ready.clear()
        log.info("[edsdk] live view off")

    def _on_property_event(self, in_event, in_property_id, in_param, in_context):
        import edsdk_raw as eds

        try:
            if (
                in_event == eds.kEdsPropertyEvent_PropertyChanged
                and in_property_id == eds.kEdsPropID_Evf_OutputDevice
            ):
                self._evf_device_ready.set()
        except Exception:
            pass
        return eds.EDS_ERR_OK

    def _on_object_event(self, in_event, in_ref, in_context):
        """WINFUNCTYPE 回调：必须尽量短，实际下载在此同步完成（与官方 Sample 一致）。"""
        import edsdk_raw as eds

        try:
            if in_event == eds.kEdsObjectEvent_DirItemRequestTransfer:
                self._download_item(in_ref)
            elif in_ref:
                self._lib.EdsRelease(in_ref)
        except Exception as e:
            self._download_error = str(e)
            log.exception("[edsdk] object event")
            self._download_done.set()
        return eds.EDS_ERR_OK

    def _download_item(self, dir_item) -> None:
        import edsdk_raw as eds
        from ctypes import c_ubyte, c_ulonglong, c_void_p

        info = eds.EdsDirectoryItemInfo()
        err = self._lib.EdsGetDirectoryItemInfo(dir_item, byref(info))
        if err != eds.EDS_ERR_OK:
            self._lib.EdsDownloadCancel(dir_item)
            self._lib.EdsRelease(dir_item)
            self._download_error = f"GetDirectoryItemInfo 0x{err:08X}"
            self._download_done.set()
            return

        cam_name = info.szFileName.decode("utf-8", errors="replace")
        dest = self._pending_dest
        if dest is None:
            dest = Path.cwd() / cam_name
        out = dest.with_name(dest.stem + Path(cam_name).suffix)
        out.parent.mkdir(parents=True, exist_ok=True)

        stream = eds.EdsStreamRef()
        # 内存流更稳（避开路径编码）；失败再回退文件流
        err = self._lib.EdsCreateMemoryStream(c_ulonglong(info.size or 1), byref(stream))
        use_mem = err == eds.EDS_ERR_OK
        if not use_mem:
            path_bytes = str(out).encode("mbcs", errors="replace")
            err = self._lib.EdsCreateFileStream(
                path_bytes,
                eds.kEdsFileCreateDisposition_CreateAlways,
                eds.kEdsAccess_ReadWrite,
                byref(stream),
            )
            if err != eds.EDS_ERR_OK:
                self._lib.EdsDownloadCancel(dir_item)
                self._lib.EdsRelease(dir_item)
                self._download_error = f"CreateFileStream 0x{err:08X} path={out}"
                self._download_done.set()
                return

        err = self._lib.EdsDownload(dir_item, info.size, stream)
        if err == eds.EDS_ERR_OK:
            self._lib.EdsDownloadComplete(dir_item)
        else:
            self._lib.EdsDownloadCancel(dir_item)
            self._download_error = f"EdsDownload 0x{err:08X}"

        if err == eds.EDS_ERR_OK and use_mem:
            ptr = c_void_p()
            length = c_ulonglong(0)
            e1 = self._lib.EdsGetPointer(stream, byref(ptr))
            e2 = self._lib.EdsGetLength(stream, byref(length))
            if e1 != eds.EDS_ERR_OK or e2 != eds.EDS_ERR_OK or not ptr.value:
                self._download_error = f"GetPointer/Length 0x{e1:08X}/0x{e2:08X}"
                err = e1 or e2
            else:
                n = int(length.value) if length.value else int(info.size)
                buf = (c_ubyte * n).from_address(ptr.value)
                out.write_bytes(bytes(buf))

        self._lib.EdsRelease(stream)
        self._lib.EdsRelease(dir_item)

        if err == eds.EDS_ERR_OK and not self._download_error:
            self._last_saved = out
            log.info("[edsdk] downloaded %s (%d bytes)", out.name, info.size)
        self._download_done.set()

def create_camera(backend: str = "stub", **kwargs) -> CameraBase:
    backend = (backend or "stub").lower().strip()
    if backend in ("stub", "dry", "dummy"):
        return StubCamera(
            **{
                k: v
                for k, v in kwargs.items()
                if k in ("af_delay_s", "shoot_delay_s", "write_placeholder")
            }
        )
    if backend in ("edsdk", "canon"):
        return EdsdkCamera(
            **{
                k: v
                for k, v in kwargs.items()
                if k
                in (
                    "library_path",
                    "download_timeout_s",
                    "keep_liveview",
                    "shutter_non_af",
                    "shoot_retries",
                    "post_download_pump_s",
                )
            }
        )
    raise ValueError(f"未知 camera_backend: {backend}")
