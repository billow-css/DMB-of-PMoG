#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment/markers.py
=====================
屏摄四角 ArUco + 粗白边框：显示合成与批量检测共用。

角点 ID（DICT_4X4_50）
----------------------
  0 = 左上 TL    1 = 右上 TR
  3 = 左下 BL    2 = 右下 BR

标记放在宿主图外侧；用各标记「朝向内容」的那一角估计宿主四角，再透视到 out_size。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# 固定字典与 ID，检测端必须一致
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
MARKER_TL, MARKER_TR, MARKER_BR, MARKER_BL = 0, 1, 2, 3
REQUIRED_IDS = (MARKER_TL, MARKER_TR, MARKER_BR, MARKER_BL)


@dataclass
class BoardLayout:
    """一帧合成画面的几何（像素，相对 board）。"""

    board_w: int
    board_h: int
    content_x0: int
    content_y0: int
    content_x1: int
    content_y1: int
    marker_size: int
    gap: int
    border: int


def _get_dictionary():
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)


def _get_detector():
    dictionary = _get_dictionary()
    # OpenCV 4.7+ Detector；旧版回退 detectMarkers
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        # 抗模糊：放宽阈值 / 周长 / 多边形近似
        if hasattr(params, "adaptiveThreshWinSizeMin"):
            params.adaptiveThreshWinSizeMin = 3
            params.adaptiveThreshWinSizeMax = 73
            params.adaptiveThreshWinSizeStep = 4
        if hasattr(params, "minMarkerPerimeterRate"):
            params.minMarkerPerimeterRate = 0.01
        if hasattr(params, "maxMarkerPerimeterRate"):
            params.maxMarkerPerimeterRate = 4.0
        if hasattr(params, "polygonalApproxAccuracyRate"):
            params.polygonalApproxAccuracyRate = 0.08
        if hasattr(params, "minCornerDistanceRate"):
            params.minCornerDistanceRate = 0.02
        if hasattr(params, "minDistanceToBorder"):
            params.minDistanceToBorder = 1
        if hasattr(params, "cornerRefinementMethod"):
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        return cv2.aruco.ArucoDetector(dictionary, params), True
    return dictionary, False


_MARKER_CACHE: dict = {}


def make_marker_bgr(marker_id: int, size: int) -> np.ndarray:
    key = (int(marker_id), int(size))
    cached = _MARKER_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    dictionary = _get_dictionary()
    # drawMarker → 单通道；转 BGR
    if hasattr(cv2.aruco, "generateImageMarker"):
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    else:
        img = np.zeros((size, size), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, marker_id, size, img, 1)
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _MARKER_CACHE[key] = out
    return out.copy()


def compose_board_bgr(
    content_rgb: np.ndarray,
    board_w: int,
    board_h: int,
    *,
    marker_size: int = 0,
    gap: int = 0,
    border: int = 0,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    frame_color: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[np.ndarray, BoardLayout]:
    """
    将 RGB 宿主图放入 board（BGR 输出），四周粗白框 + 四角 ArUco。

    参数为 0 时按 board 短边自动比例选取，保证拍屏后标记仍够大。
    """
    if content_rgb.ndim != 3 or content_rgb.shape[2] != 3:
        raise ValueError("content_rgb 需为 HxWx3")

    short = min(board_w, board_h)
    if marker_size <= 0:
        marker_size = max(48, int(short * 0.10))  # 约短边 10%
    if gap <= 0:
        gap = max(8, int(marker_size * 0.15))
    if border <= 0:
        border = max(6, int(short * 0.012))

    # 内容区：去掉边框、标记带、间隙后剩余区域，等比放入
    band = border + gap + marker_size + gap
    inner_w = board_w - 2 * band
    inner_h = board_h - 2 * band
    if inner_w < 32 or inner_h < 32:
        # 屏幕过小：缩小标记
        marker_size = max(32, short // 16)
        gap = max(4, marker_size // 8)
        border = max(4, short // 80)
        band = border + gap + marker_size + gap
        inner_w = board_w - 2 * band
        inner_h = board_h - 2 * band
    if inner_w < 16 or inner_h < 16:
        raise ValueError(f"board 太小无法放置标记: {board_w}x{board_h}")

    ch, cw = content_rgb.shape[:2]
    scale = min(inner_w / cw, inner_h / ch)
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    content_bgr = cv2.cvtColor(content_rgb, cv2.COLOR_RGB2BGR)
    content_bgr = cv2.resize(content_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    board = np.zeros((board_h, board_w, 3), dtype=np.uint8)
    board[:] = bg_color

    # 白边框（外框）
    cv2.rectangle(
        board,
        (border // 2, border // 2),
        (board_w - 1 - border // 2, board_h - 1 - border // 2),
        frame_color,
        thickness=max(2, border),
    )

    cx0 = (board_w - nw) // 2
    cy0 = (board_h - nh) // 2
    # 保证内容不压到标记带
    cx0 = int(np.clip(cx0, band, board_w - band - nw))
    cy0 = int(np.clip(cy0, band, board_h - band - nh))
    cx1, cy1 = cx0 + nw, cy0 + nh
    board[cy0:cy1, cx0:cx1] = content_bgr

    # 内容区再描一圈细白边，便于轮廓兜底
    cv2.rectangle(board, (cx0 - 2, cy0 - 2), (cx1 + 1, cy1 + 1), frame_color, 2)

    s = marker_size
    g = gap
    # 标记贴在内容四角外侧
    positions = {
        MARKER_TL: (cx0 - g - s, cy0 - g - s),
        MARKER_TR: (cx1 + g, cy0 - g - s),
        MARKER_BR: (cx1 + g, cy1 + g),
        MARKER_BL: (cx0 - g - s, cy1 + g),
    }
    for mid, (mx, my) in positions.items():
        mx = int(np.clip(mx, border, board_w - s - border))
        my = int(np.clip(my, border, board_h - s - border))
        marker = make_marker_bgr(mid, s)
        # 白底安静区
        pad = max(2, s // 16)
        x0, y0 = mx - pad, my - pad
        x1, y1 = mx + s + pad, my + s + pad
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(board_w, x1), min(board_h, y1)
        board[y0:y1, x0:x1] = 255
        board[my : my + s, mx : mx + s] = marker

    layout = BoardLayout(
        board_w=board_w,
        board_h=board_h,
        content_x0=cx0,
        content_y0=cy0,
        content_x1=cx1,
        content_y1=cy1,
        marker_size=s,
        gap=g,
        border=border,
    )
    return board, layout


def _detect_aruco(gray: np.ndarray):
    detector, is_new = _get_detector()
    if is_new:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, detector)
    return corners, ids


def _collect_partial_corners(
    corners: Sequence, ids: np.ndarray
) -> Dict[int, np.ndarray]:
    """各标记 → 内容角候选（不一定齐全）。"""
    id_to_corners: Dict[int, np.ndarray] = {}
    for i, mid in enumerate(ids.flatten().tolist()):
        id_to_corners[int(mid)] = corners[i].reshape(4, 2)

    # 标记角点顺序：0=TL,1=TR,2=BR,3=BL；取朝向内容的内角
    inner_idx = {
        MARKER_TL: 2,
        MARKER_TR: 3,
        MARKER_BR: 0,
        MARKER_BL: 1,
    }
    partial: Dict[int, np.ndarray] = {}
    for mid, ci in inner_idx.items():
        if mid in id_to_corners:
            partial[mid] = id_to_corners[mid][ci].astype(np.float32)
    return partial


def _complete_quad_from_partial(
    partial: Dict[int, np.ndarray],
) -> Tuple[Optional[np.ndarray], str]:
    """
    用平行四边形 / 轴对齐假设补全缺失角。

    返回 (quad TL,TR,BR,BL 或 None, method)
    """
    have = set(partial.keys())
    pts = {k: v.copy() for k, v in partial.items()}

    # 平行四边形：TL + BR = TR + BL
    def fill_para() -> bool:
        changed = False
        if MARKER_TL in pts and MARKER_TR in pts and MARKER_BR in pts and MARKER_BL not in pts:
            pts[MARKER_BL] = pts[MARKER_TL] + pts[MARKER_BR] - pts[MARKER_TR]
            changed = True
        if MARKER_TL in pts and MARKER_TR in pts and MARKER_BL in pts and MARKER_BR not in pts:
            pts[MARKER_BR] = pts[MARKER_TR] + pts[MARKER_BL] - pts[MARKER_TL]
            changed = True
        if MARKER_TL in pts and MARKER_BR in pts and MARKER_BL in pts and MARKER_TR not in pts:
            pts[MARKER_TR] = pts[MARKER_TL] + pts[MARKER_BR] - pts[MARKER_BL]
            changed = True
        if MARKER_TR in pts and MARKER_BR in pts and MARKER_BL in pts and MARKER_TL not in pts:
            pts[MARKER_TL] = pts[MARKER_TR] + pts[MARKER_BL] - pts[MARKER_BR]
            changed = True
        return changed

    method = "full"
    if len(have) == 4:
        method = "full"
    elif len(have) == 3:
        fill_para()
        method = "para3"
    elif len(have) == 2:
        # 对角：轴对齐补全
        if have == {MARKER_TL, MARKER_BR}:
            tl, br = pts[MARKER_TL], pts[MARKER_BR]
            pts[MARKER_TR] = np.array([br[0], tl[1]], dtype=np.float32)
            pts[MARKER_BL] = np.array([tl[0], br[1]], dtype=np.float32)
            method = "diag_axis"
        elif have == {MARKER_TR, MARKER_BL}:
            tr, bl = pts[MARKER_TR], pts[MARKER_BL]
            pts[MARKER_TL] = np.array([bl[0], tr[1]], dtype=np.float32)
            pts[MARKER_BR] = np.array([tr[0], bl[1]], dtype=np.float32)
            method = "diag_axis"
        else:
            # 邻边：用两边向量估计对边（假设矩形透视较弱）
            if have == {MARKER_TL, MARKER_TR}:
                tl, tr = pts[MARKER_TL], pts[MARKER_TR]
                edge = tr - tl
                # 向下偏移：用边长估计高度
                side = float(np.linalg.norm(edge))
                down = np.array([0.0, side], dtype=np.float32)
                pts[MARKER_BL] = tl + down
                pts[MARKER_BR] = tr + down
                method = "edge_down"
            elif have == {MARKER_BL, MARKER_BR}:
                bl, br = pts[MARKER_BL], pts[MARKER_BR]
                edge = br - bl
                side = float(np.linalg.norm(edge))
                up = np.array([0.0, -side], dtype=np.float32)
                pts[MARKER_TL] = bl + up
                pts[MARKER_TR] = br + up
                method = "edge_up"
            elif have == {MARKER_TL, MARKER_BL}:
                tl, bl = pts[MARKER_TL], pts[MARKER_BL]
                edge = bl - tl
                side = float(np.linalg.norm(edge))
                right = np.array([side, 0.0], dtype=np.float32)
                pts[MARKER_TR] = tl + right
                pts[MARKER_BR] = bl + right
                method = "edge_right"
            elif have == {MARKER_TR, MARKER_BR}:
                tr, br = pts[MARKER_TR], pts[MARKER_BR]
                edge = br - tr
                side = float(np.linalg.norm(edge))
                left = np.array([-side, 0.0], dtype=np.float32)
                pts[MARKER_TL] = tr + left
                pts[MARKER_BL] = br + left
                method = "edge_left"
            else:
                return None, "unsupported2"
    elif len(have) == 1:
        return None, "only1"
    else:
        return None, "none"

    if not all(m in pts for m in REQUIRED_IDS):
        fill_para()
    if not all(m in pts for m in REQUIRED_IDS):
        return None, method

    quad = np.array(
        [pts[MARKER_TL], pts[MARKER_TR], pts[MARKER_BR], pts[MARKER_BL]],
        dtype=np.float32,
    )
    return quad, method


def _quad_geometry_score(quad: np.ndarray, img_h: int, img_w: int) -> float:
    """几何合理性 0~1：凸、面积、长宽比。"""
    if quad is None or quad.shape != (4, 2):
        return 0.0
    margin = 0.08
    for x, y in quad:
        if x < -margin * img_w or x > (1 + margin) * img_w:
            return 0.0
        if y < -margin * img_h or y > (1 + margin) * img_h:
            return 0.0

    q = quad.astype(np.float32)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    signs = [cross(q[i], q[(i + 1) % 4], q[(i + 2) % 4]) for i in range(4)]
    if any(abs(s) < 1e-6 for s in signs):
        return 0.1
    if not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
        return 0.0

    area = abs(cv2.contourArea(q))
    img_area = float(img_h * img_w)
    ar = area / max(img_area, 1.0)
    if ar < 0.02 or ar > 0.95:
        return 0.15

    edges = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
    if min(edges) < 1e-3:
        return 0.0
    ratio = max(edges) / min(edges)
    if ratio > 8:
        return 0.2

    area_s = float(np.clip((ar - 0.02) / 0.4, 0.0, 1.0))
    ratio_s = float(np.clip(1.0 - (ratio - 1.0) / 7.0, 0.0, 1.0))
    return 0.45 * area_s + 0.55 * ratio_s


def estimate_confidence(
    n_markers: int,
    method: str,
    geo: float,
    *,
    fallback: str = "",
) -> float:
    """
    综合置信度 0~1。

    - 4 角齐全 ≈ 0.85–1.0
    - 3 角补全 ≈ 0.65–0.85
    - 2 角估计 ≈ 0.45–0.70
    - 仅白框 ≈ 0.40–0.65
    """
    base = {
        "full": 0.92,
        "para3": 0.72,
        "diag_axis": 0.58,
        "edge_down": 0.52,
        "edge_up": 0.52,
        "edge_right": 0.52,
        "edge_left": 0.52,
        "white_frame": 0.45,
        "white_pads": 0.55,  # 糊到解不出 ArUco，四角白垫（优先于整屏白框）
    }.get(method, 0.40)

    if fallback == "white_frame" and method != "white_frame":
        base = max(base, 0.50)
    if n_markers >= 4:
        base = max(base, 0.90)
    elif n_markers == 3:
        base = max(base, 0.68)
    elif n_markers == 2:
        base = min(base, 0.65)

    # 与几何分融合
    conf = 0.55 * base + 0.45 * (0.3 + 0.7 * geo)
    return float(np.clip(conf, 0.0, 1.0))


def detect_content_quad(
    bgr: np.ndarray,
    *,
    scales: Sequence[float] = (1.0, 1.5, 0.75, 2.0, 2.5),
    min_confidence: float = 0.25,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    多尺度 ArUco 检测内容四边形；允许缺角补全。

    当 ``confidence >= min_confidence`` 时返回 quad，否则 None。
    默认阈值较低：运动模糊时宁可粗截，也不轻易失败。
    """
    if bgr is None or bgr.size == 0:
        return None, {"ok": False, "reason": "empty", "confidence": 0.0}

    h, w = bgr.shape[:2]
    gray0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants = [
        ("raw", gray0),
        ("clahe", clahe.apply(gray0)),
        ("sharp", cv2.addWeighted(gray0, 1.6, cv2.GaussianBlur(gray0, (0, 0), 1.5), -0.6, 0)),
        ("eq", cv2.equalizeHist(gray0)),
    ]

    candidates: List[Tuple[np.ndarray, dict]] = []

    for vname, gray in variants:
        for sc in scales:
            if abs(sc - 1.0) < 1e-6:
                g = gray
            else:
                g = cv2.resize(
                    gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR
                )
            corners, ids = _detect_aruco(g)
            if ids is None or len(ids) == 0:
                continue
            if abs(sc - 1.0) > 1e-6:
                corners = [c / sc for c in corners]

            found = sorted(int(x) for x in ids.flatten())
            n_req = sum(1 for m in REQUIRED_IDS if m in found)
            if n_req < 2:
                continue

            partial = _collect_partial_corners(corners, ids)
            # 只保留四个标准 ID 的内容角
            partial = {k: v for k, v in partial.items() if k in REQUIRED_IDS}
            quad, method = _complete_quad_from_partial(partial)
            if quad is None:
                continue
            geo = _quad_geometry_score(quad, h, w)
            conf = estimate_confidence(n_req, method, geo)
            meta = {
                "ok": conf >= min_confidence,
                "variant": vname,
                "scale": sc,
                "ids": found,
                "n_required": n_req,
                "method": method,
                "geo": round(geo, 3),
                "confidence": round(conf, 3),
            }
            candidates.append((quad, meta))
            if n_req == 4 and conf >= 0.85:
                return quad, meta

    # 糊到解不出码：用四角白色标记垫估计内容框（可不严格）
    quad_pads, meta_pads = _detect_white_pad_quad(bgr)
    if quad_pads is not None:
        quad_pads = _shrink_quad(quad_pads, 0.90)
        geo = _quad_geometry_score(quad_pads, h, w)
        conf = estimate_confidence(0, "white_pads", geo, fallback="white_pads")
        if geo >= 0.30:
            conf = max(conf, 0.40)
        meta_pads.update(
            {
                "ok": conf >= min_confidence,
                "fallback": "white_pads",
                "method": "white_pads",
                "geo": round(geo, 3),
                "confidence": round(conf, 3),
            }
        )
        candidates.append((quad_pads, meta_pads))

    # 白框兜底（偏外，再向内收一圈）
    quad2, meta2 = _detect_white_frame_quad(bgr)
    if quad2 is not None:
        quad2 = _shrink_quad(quad2, 0.82)
        geo = _quad_geometry_score(quad2, h, w)
        conf = estimate_confidence(0, "white_frame", geo, fallback="white_frame")
        if geo >= 0.30:
            conf = max(conf, 0.32)
        meta2.update(
            {
                "ok": conf >= min_confidence,
                "fallback": "white_frame",
                "method": "white_frame",
                "n_required": 0,
                "ids": [],
                "geo": round(geo, 3),
                "confidence": round(conf, 3),
            }
        )
        candidates.append((quad2, meta2))

    if not candidates:
        return None, {"ok": False, "reason": "no_markers", "confidence": 0.0}

    # 选置信最高
    candidates.sort(key=lambda x: x[1].get("confidence", 0.0), reverse=True)
    best_quad, best_meta = candidates[0]
    if best_meta.get("confidence", 0.0) >= min_confidence:
        best_meta["ok"] = True
        return best_quad, best_meta

    best_meta["ok"] = False
    best_meta["reason"] = (
        f"low_confidence {best_meta.get('confidence')} < {min_confidence}"
    )
    return None, best_meta


def _order_quad_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _shrink_quad(quad: np.ndarray, scale: float) -> np.ndarray:
    """相对中心缩放四边形（scale<1 向内收，粗截时去掉外围标记带）。"""
    q = quad.astype(np.float32).reshape(4, 2)
    c = q.mean(axis=0)
    return (c + (q - c) * float(scale)).astype(np.float32)


def _detect_white_pad_quad(bgr: np.ndarray) -> Tuple[Optional[np.ndarray], dict]:
    """
    运动模糊时 ArUco 解不出码，改找四角「白底标记垫」估计内容框。

    截取不必严格：用各垫朝向中心的内角，略包住宿主图即可。
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    img_area = float(h * w)
    best_quad: Optional[np.ndarray] = None
    best_meta: dict = {"ok": False, "reason": "no_pads"}
    best_geo = -1.0

    for thr in (170, 185, 200, 215, 230):
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, th = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pads: List[Tuple[float, float, float, int, int, int, int]] = []
        for c in cnts:
            area = float(cv2.contourArea(c))
            # 标记垫：比高光内容块小，比噪点大
            if area < 0.0002 * img_area or area > 0.012 * img_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            ar = bw / max(bh, 1)
            if ar < 0.35 or ar > 2.8:
                continue
            fill = area / float(bw * bh + 1e-6)
            if fill < 0.30:
                continue
            pads.append((x + bw * 0.5, y + bh * 0.5, area, x, y, bw, bh))

        if len(pads) < 4:
            continue

        # 按面积聚一类（四角标记尺寸接近）
        pads_sorted = sorted(pads, key=lambda t: t[2], reverse=True)
        for ref in pads_sorted[:8]:
            group = [p for p in pads if 0.35 * ref[2] <= p[2] <= 2.8 * ref[2]]
            if len(group) < 4:
                continue

            xs = [p[0] for p in group]
            ys = [p[1] for p in group]
            mx, my = float(np.median(xs)), float(np.median(ys))

            tl = [p for p in group if p[0] <= mx and p[1] <= my]
            tr = [p for p in group if p[0] > mx and p[1] <= my]
            bl = [p for p in group if p[0] <= mx and p[1] > my]
            br = [p for p in group if p[0] > mx and p[1] > my]
            if not (tl and tr and bl and br):
                continue

            def farthest(bucket):
                return max(bucket, key=lambda p: (p[0] - mx) ** 2 + (p[1] - my) ** 2)

            p_tl, p_tr, p_br, p_bl = (
                farthest(tl),
                farthest(tr),
                farthest(br),
                farthest(bl),
            )

            # 各垫「朝向内容」的内角（不严格，允许多包一点标记）
            def inner(pad, which: str) -> np.ndarray:
                _, _, _, x, y, bw, bh = pad
                inset = 0.15
                if which == "tl":
                    return np.array(
                        [x + bw * (1 - inset), y + bh * (1 - inset)], dtype=np.float32
                    )
                if which == "tr":
                    return np.array(
                        [x + bw * inset, y + bh * (1 - inset)], dtype=np.float32
                    )
                if which == "br":
                    return np.array(
                        [x + bw * inset, y + bh * inset], dtype=np.float32
                    )
                return np.array(
                    [x + bw * (1 - inset), y + bh * inset], dtype=np.float32
                )

            quad = np.array(
                [
                    inner(p_tl, "tl"),
                    inner(p_tr, "tr"),
                    inner(p_br, "br"),
                    inner(p_bl, "bl"),
                ],
                dtype=np.float32,
            )
            geo = _quad_geometry_score(quad, h, w)
            if geo > best_geo:
                best_geo = geo
                best_quad = quad
                best_meta = {
                    "ok": True,
                    "n_required": 0,
                    "ids": [],
                    "thr": thr,
                    "n_pads": len(group),
                }

    if best_quad is None:
        return None, best_meta
    return best_quad, best_meta


def _detect_white_frame_quad(bgr: np.ndarray) -> Tuple[Optional[np.ndarray], dict]:
    """亮边框兜底（次优，糊得很时可碰运气）。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    img_area = float(h * w)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    for thr_mode in ("otsu", 180, 200, 220):
        if thr_mode == "otsu":
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, th = cv2.threshold(blur, int(thr_mode), 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts[:12]:
            area = cv2.contourArea(c)
            if area < 0.04 * img_area or area > 0.98 * img_area:
                continue
            peri = cv2.arcLength(c, True)
            for eps in (0.02, 0.04, 0.06):
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    quad = _order_quad_tl_tr_br_bl(approx)
                    return quad, {"ok": True, "n_required": 0, "ids": []}
            # 凸包四点
            hull = cv2.convexHull(c)
            if len(hull) >= 4:
                peri_h = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.05 * peri_h, True)
                if len(approx) == 4:
                    quad = _order_quad_tl_tr_br_bl(approx)
                    return quad, {"ok": True, "n_required": 0, "ids": []}
    return None, {"ok": False, "reason": "no_quad"}


def warp_content(
    bgr: np.ndarray,
    quad: np.ndarray,
    out_size: int = 128,
) -> np.ndarray:
    """透视到 out_size×out_size。"""
    dst = np.array(
        [
            [0, 0],
            [out_size - 1, 0],
            [out_size - 1, out_size - 1],
            [0, out_size - 1],
        ],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(bgr, m, (out_size, out_size), flags=cv2.INTER_LINEAR)


def draw_debug(bgr: np.ndarray, quad: Optional[np.ndarray], info: dict) -> np.ndarray:
    vis = bgr.copy()
    if quad is not None:
        q = quad.astype(np.int32)
        cv2.polylines(vis, [q], True, (0, 255, 0), 3)
        for i, (x, y) in enumerate(q):
            cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 255), -1)
            cv2.putText(
                vis,
                str(i),
                (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )
    label = (
        f"ids={info.get('ids')} n={info.get('n_required')} "
        f"conf={info.get('confidence')} method={info.get('method')} "
        f"scale={info.get('scale')} {info.get('variant', '')}"
    )
    cv2.putText(
        vis, label[:90], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )
    return vis
