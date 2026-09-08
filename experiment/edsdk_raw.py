#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment/edsdk_raw.py
=======================
Canon EDSDK ctypes 绑定（精简版，覆盖遥控拍照 + 回传）。

SDK 来源：``EDSDK132011CD(13.20.21)/Windows/EDSDK_64``
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_char,
    c_char_p,
    c_int,
    c_uint,
    c_ulonglong,
    c_void_p,
)
from pathlib import Path
from typing import Optional

# ---- constants (from EDSDKTypes.h / EDSDKErrors.h) ----
EDS_ERR_OK = 0x00000000
EDS_ERR_DEVICE_BUSY = 0x00000081
EDS_ERR_OBJECT_NOTREADY = 0x0000A102
EDS_MAX_NAME = 256

kEdsPropID_ProductName = 0x00000002
kEdsPropID_SaveTo = 0x0000000B
# 注意：与 EDSDKTypes.h 一致 — OutputDevice=0x500，Mode=0x501（勿对调）
kEdsPropID_Evf_OutputDevice = 0x00000500
kEdsPropID_Evf_Mode = 0x00000501
kEdsPropID_Evf_ImagePosition = 0x0000050B

kEdsCameraCommand_DoEvfAf = 0x00000102
kEdsCameraCommand_PressShutterButton = 0x00000004

kEdsCameraCommand_EvfAf_OFF = 0
kEdsCameraCommand_EvfAf_ON = 1

kEdsCameraCommand_ShutterButton_OFF = 0x00000000
kEdsCameraCommand_ShutterButton_Halfway = 0x00000001
kEdsCameraCommand_ShutterButton_Completely = 0x00000003
kEdsCameraCommand_ShutterButton_Halfway_NonAF = 0x00010001
kEdsCameraCommand_ShutterButton_Completely_NonAF = 0x00010003

kEdsCameraCommand_ExtendShutDownTimer = 0x00000001

EDS_ERR_TAKE_PICTURE_AF_NG = 0x00008D01

kEdsSaveTo_Camera = 1
kEdsSaveTo_Host = 2
kEdsSaveTo_Both = 3

kEdsEvfOutputDevice_TFT = 1
kEdsEvfOutputDevice_PC = 2
kEdsEvfOutputDevice_PC_Small = 8

kEdsPropertyEvent_All = 0x00000100
kEdsPropertyEvent_PropertyChanged = 0x00000101

kEdsObjectEvent_DirItemRequestTransfer = 0x00000208
kEdsObjectEvent_All = 0x00000200

kEdsFileCreateDisposition_CreateNew = 0
kEdsFileCreateDisposition_CreateAlways = 1
kEdsAccess_Read = 0
kEdsAccess_Write = 1
kEdsAccess_ReadWrite = 2  # 勿写成 3，否则 CreateFileStream → 0xAB STREAM_BAD_OPTIONS

kEdsSeek_Cur = 0
kEdsSeek_Begin = 1
kEdsSeek_End = 2


EdsBaseRef = c_void_p
EdsCameraRef = c_void_p
EdsCameraListRef = c_void_p
EdsDirectoryItemRef = c_void_p
EdsStreamRef = c_void_p
EdsEvfImageRef = c_void_p
EdsError = c_uint


class EdsDeviceInfo(Structure):
    _fields_ = [
        ("szPortName", c_char * EDS_MAX_NAME),
        ("szDeviceDescription", c_char * EDS_MAX_NAME),
        ("deviceSubType", c_uint),
        ("reserved", c_uint),
    ]


class EdsDirectoryItemInfo(Structure):
    _fields_ = [
        ("size", c_ulonglong),
        ("isFolder", c_int),
        ("groupID", c_uint),
        ("option", c_uint),
        ("szFileName", c_char * EDS_MAX_NAME),
        ("format", c_uint),
        ("dateTime", c_uint),
    ]


class EdsCapacity(Structure):
    _fields_ = [
        ("numberOfFreeClusters", c_int),
        ("bytesPerSector", c_int),
        ("reset", c_int),
    ]


OBJECT_EVENT_HANDLER = ctypes.WINFUNCTYPE(
    EdsError, c_uint, EdsBaseRef, c_void_p
)
PROPERTY_EVENT_HANDLER = ctypes.WINFUNCTYPE(
    EdsError, c_uint, c_uint, c_uint, c_void_p
)


def default_edsdk_dir() -> Path:
    here = Path(__file__).resolve().parent
    root = here / "EDSDK132011CD(13.20.21)" / "Windows"
    if sys.maxsize > 2**32:
        return root / "EDSDK_64" / "Dll"
    return root / "EDSDK" / "Dll"


def load_edsdk(dll_dir: Optional[str | Path] = None) -> ctypes.WinDLL:
    """加载 EDSDK.dll，并把目录加入 PATH 以便 EdsImage.dll 可被找到。"""
    dll_dir = Path(dll_dir) if dll_dir else default_edsdk_dir()
    if not dll_dir.is_dir():
        raise FileNotFoundError(f"EDSDK Dll 目录不存在: {dll_dir}")
    dll_path = dll_dir / "EDSDK.dll"
    if not dll_path.is_file():
        raise FileNotFoundError(f"找不到 EDSDK.dll: {dll_path}")

    # Windows: 依赖 DLL 同目录搜索
    os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(dll_dir))

    lib = ctypes.WinDLL(str(dll_path))

    def _err(name, restype, *argtypes):
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = list(argtypes)
        return fn

    lib.EdsInitializeSDK = _err("EdsInitializeSDK", EdsError)
    lib.EdsTerminateSDK = _err("EdsTerminateSDK", EdsError)
    lib.EdsGetCameraList = _err("EdsGetCameraList", EdsError, POINTER(EdsCameraListRef))
    lib.EdsGetChildCount = _err(
        "EdsGetChildCount", EdsError, EdsBaseRef, POINTER(c_int)
    )
    lib.EdsGetChildAtIndex = _err(
        "EdsGetChildAtIndex",
        EdsError,
        EdsBaseRef,
        c_int,
        POINTER(EdsBaseRef),
    )
    lib.EdsGetDeviceInfo = _err(
        "EdsGetDeviceInfo", EdsError, EdsCameraRef, POINTER(EdsDeviceInfo)
    )
    lib.EdsOpenSession = _err("EdsOpenSession", EdsError, EdsCameraRef)
    lib.EdsCloseSession = _err("EdsCloseSession", EdsError, EdsCameraRef)
    lib.EdsRetain = _err("EdsRetain", c_uint, EdsBaseRef)
    lib.EdsRelease = _err("EdsRelease", c_uint, EdsBaseRef)
    lib.EdsSendCommand = _err(
        "EdsSendCommand", EdsError, EdsCameraRef, c_uint, c_int
    )
    lib.EdsSetPropertyData = _err(
        "EdsSetPropertyData",
        EdsError,
        EdsBaseRef,
        c_uint,
        c_int,
        c_int,
        c_void_p,
    )
    lib.EdsGetPropertyData = _err(
        "EdsGetPropertyData",
        EdsError,
        EdsBaseRef,
        c_uint,
        c_int,
        c_int,
        c_void_p,
    )
    lib.EdsSetCapacity = _err(
        "EdsSetCapacity", EdsError, EdsCameraRef, EdsCapacity
    )
    lib.EdsSetObjectEventHandler = _err(
        "EdsSetObjectEventHandler",
        EdsError,
        EdsCameraRef,
        c_uint,
        OBJECT_EVENT_HANDLER,
        c_void_p,
    )
    lib.EdsSetPropertyEventHandler = _err(
        "EdsSetPropertyEventHandler",
        EdsError,
        EdsCameraRef,
        c_uint,
        PROPERTY_EVENT_HANDLER,
        c_void_p,
    )
    lib.EdsGetEvent = _err("EdsGetEvent", EdsError)
    lib.EdsSeek = _err(
        "EdsSeek", EdsError, EdsStreamRef, c_int, c_uint
    )
    lib.EdsGetDirectoryItemInfo = _err(
        "EdsGetDirectoryItemInfo",
        EdsError,
        EdsDirectoryItemRef,
        POINTER(EdsDirectoryItemInfo),
    )
    lib.EdsCreateFileStream = _err(
        "EdsCreateFileStream",
        EdsError,
        c_char_p,
        c_int,
        c_int,
        POINTER(EdsStreamRef),
    )
    lib.EdsCreateMemoryStream = _err(
        "EdsCreateMemoryStream",
        EdsError,
        c_ulonglong,
        POINTER(EdsStreamRef),
    )
    lib.EdsCreateMemoryStreamFromPointer = _err(
        "EdsCreateMemoryStreamFromPointer",
        EdsError,
        c_void_p,
        c_ulonglong,
        POINTER(EdsStreamRef),
    )
    lib.EdsGetPointer = _err(
        "EdsGetPointer", EdsError, EdsStreamRef, POINTER(c_void_p)
    )
    lib.EdsGetLength = _err(
        "EdsGetLength", EdsError, EdsStreamRef, POINTER(c_ulonglong)
    )
    lib.EdsDownload = _err(
        "EdsDownload", EdsError, EdsDirectoryItemRef, c_ulonglong, EdsStreamRef
    )
    lib.EdsDownloadComplete = _err(
        "EdsDownloadComplete", EdsError, EdsDirectoryItemRef
    )
    lib.EdsDownloadCancel = _err(
        "EdsDownloadCancel", EdsError, EdsDirectoryItemRef
    )
    lib.EdsCreateEvfImageRef = _err(
        "EdsCreateEvfImageRef",
        EdsError,
        EdsStreamRef,
        POINTER(EdsEvfImageRef),
    )
    lib.EdsDownloadEvfImage = _err(
        "EdsDownloadEvfImage",
        EdsError,
        EdsCameraRef,
        EdsEvfImageRef,
    )
    return lib


def check(err: int, what: str) -> None:
    if err != EDS_ERR_OK:
        raise RuntimeError(f"EDSDK {what} failed: 0x{err:08X}")


def retry_busy(fn, what: str, retries: int = 20, delay_s: float = 0.1) -> None:
    import time

    last = 0
    for _ in range(retries):
        last = int(fn())
        if last == EDS_ERR_OK:
            return
        if last == EDS_ERR_DEVICE_BUSY:
            time.sleep(delay_s)
            continue
        raise RuntimeError(f"EDSDK {what} failed: 0x{last:08X}")
    raise RuntimeError(f"EDSDK {what} still busy: 0x{last:08X}")
