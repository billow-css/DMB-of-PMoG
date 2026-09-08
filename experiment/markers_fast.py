#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment/markers_fast.py
==========================
快速 + 稳妥 的内容框检测（供 ``batch_rectify.py`` / ``rectify_gui`` 使用）。

权衡策略（抖动屏摄）：
  - 内容糊可以，四边形几何不能歪 → 畸形直接判失败，进 failed/
  - 优先 ArUco（缩略图上多试几次，仍远快于全图穷举）
  - 白垫 / 白框仅在几何很像矩形时才接受，且置信门槛更高
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from markers import (
    REQUIRED_IDS,
    _collect_partial_corners,
    _complete_quad_from_partial,
    _order_quad_tl_tr_br_bl,
    _quad_geometry_score,
    _shrink_quad,
    draw_debug,
    estimate_confidence,
    warp_content,
)

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50


def _fast_detector(relaxed: bool = False):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        if hasattr(params, "adaptiveThreshWinSizeMin"):
            params.adaptiveThreshWinSizeMin = 3
            params.adaptiveThreshWinSizeMax = 35 if relaxed else 23
            params.adaptiveThreshWinSizeStep = 8 if relaxed else 10
        if hasattr(params, "minMarkerPerimeterRate"):
            params.minMarkerPerimeterRate = 0.012 if relaxed else 0.015
        if hasattr(params, "polygonalApproxAccuracyRate"):
            params.polygonalApproxAccuracyRate = 0.07 if relaxed else 0.06
        if hasattr(params, "cornerRefinementMethod"):
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        return cv2.aruco.ArucoDetector(dictionary, params), True
    return dictionary, False


_DETECTOR_CACHE = {}


def _get_fast_detector(relaxed: bool = False):
    key = "relaxed" if relaxed else "strict"
    if key not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[key] = _fast_detector(relaxed=relaxed)
    return _DETECTOR_CACHE[key]


def _detect_aruco_fast(gray: np.ndarray, relaxed: bool = False):
    detector, is_new = _get_fast_detector(relaxed=relaxed)
    if is_new:
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, detector)


def _resize_for_detect(
    bgr: np.ndarray, max_side: int
) -> Tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr, 1.0
    scale = max_side / float(m)
    small = cv2.resize(
        bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    return small, scale


def _edge_stats(quad: np.ndarray) -> Tuple[float, float, float]:
    """返回 (max/min 边长比, 对边比均值, 邻边夹角余弦绝对值均值)。"""
    q = quad.reshape(4, 2).astype(np.float64)
    edges = [q[(i + 1) % 4] - q[i] for i in range(4)]
    lengths = [float(np.linalg.norm(e)) + 1e-9 for e in edges]
    ratio_mm = max(lengths) / min(lengths)
    opp = 0.5 * (
        max(lengths[0], lengths[2]) / min(lengths[0], lengths[2])
        + max(lengths[1], lengths[3]) / min(lengths[1], lengths[3])
    )
    cos_abs = []
    for i in range(4):
        a = edges[i] / lengths[i]
        b = edges[(i + 1) % 4] / lengths[(i + 1) % 4]
        cos_abs.append(abs(float(np.dot(a, b))))
    return ratio_mm, opp, float(np.mean(cos_abs))


def _quad_sane(
    quad: np.ndarray,
    img_h: int,
    img_w: int,
    *,
    strict: bool,
) -> bool:
    """
    几何门控：抖动糊图允许，透视畸形不允许。
    strict=True 用于白垫/白框兜底。
    """
    if quad is None or quad.shape != (4, 2):
        return False
    geo = _quad_geometry_score(quad, img_h, img_w)
    if geo < (0.55 if strict else 0.35):
        return False

    q = quad.astype(np.float32)
    area = abs(cv2.contourArea(q))
    img_area = float(img_h * img_w)
    ar = area / max(img_area, 1.0)
    if ar < (0.06 if strict else 0.03) or ar > 0.85:
        return False

    ratio_mm, opp, mean_cos = _edge_stats(q)
    if strict:
        if ratio_mm > 2.2 or opp > 1.55 or mean_cos > 0.55:
            return False
    else:
        if ratio_mm > 3.5 or opp > 2.2 or mean_cos > 0.75:
            return False

    # 轴对齐包围盒长宽比不宜极端（屏摄内容近似方图）
    xs, ys = q[:, 0], q[:, 1]
    bw = float(xs.max() - xs.min()) + 1e-6
    bh = float(ys.max() - ys.min()) + 1e-6
    box_ar = max(bw, bh) / min(bw, bh)
    if box_ar > (2.4 if strict else 3.5):
        return False
    return True


def _rank(meta: dict) -> float:
    """Sort score: ArUco > pads > frame; then confidence/geo."""
    method = meta.get("method", "")
    n = int(meta.get("n_required") or 0)
    conf = float(meta.get("confidence") or 0.0)
    geo = float(meta.get("geo") or 0.0)
    base = {
        "full": 3.0,
        "para3": 2.4,
        "diag_axis": 1.6,
        "edge_down": 1.2,
        "edge_up": 1.2,
        "edge_right": 1.2,
        "edge_left": 1.2,
        "white_pads": 0.8,
        "white_frame": 0.5,
    }.get(method, 0.3)
    if n >= 4:
        base = max(base, 3.0)
    elif n == 3:
        base = max(base, 2.3)
    return base + 0.5 * conf + 0.3 * geo


def _try_aruco_on_gray(
    gray: np.ndarray,
    inv_scale: float,
    h0: int,
    w0: int,
    *,
    variant: str,
    min_markers: int = 2,
    relaxed: bool = False,
) -> Optional[Tuple[np.ndarray, dict]]:
    corners, ids, _ = _detect_aruco_fast(gray, relaxed=relaxed)
    if ids is None or len(ids) == 0:
        return None
    found = sorted(int(x) for x in ids.flatten())
    n_req = sum(1 for m in REQUIRED_IDS if m in found)
    if n_req < min_markers:
        return None
    if abs(inv_scale - 1.0) > 1e-9:
        corners = [c * inv_scale for c in corners]
    partial = _collect_partial_corners(corners, ids)
    partial = {k: v for k, v in partial.items() if k in REQUIRED_IDS}
    quad, method = _complete_quad_from_partial(partial)
    if quad is None:
        return None
    # 2 角补全更容易歪，用严格门控
    strict = n_req < 3 or method not in ("full", "para3")
    if not _quad_sane(quad, h0, w0, strict=strict):
        return None
    geo = _quad_geometry_score(quad, h0, w0)
    conf = estimate_confidence(n_req, method, geo)
    # 仅 2 角时压低分，避免压过可靠 3/4 角
    if n_req <= 2:
        conf = min(conf, 0.55)
    meta = {
        "ok": True,
        "variant": variant,
        "scale": round(1.0 / inv_scale, 3) if inv_scale else 1.0,
        "ids": found,
        "n_required": n_req,
        "method": method,
        "geo": round(geo, 3),
        "confidence": round(conf, 3),
        "fast": True,
    }
    return quad, meta


def _detect_white_pads_fast(
    bgr: np.ndarray, inv_scale: float, h0: int, w0: int
) -> Optional[Tuple[np.ndarray, dict]]:
    """四角白垫；几何不达标则返回 None（宁缺勿畸）。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    img_area = float(h * w)
    best_quad = None
    best_geo = -1.0
    best_meta: dict = {}

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    for thr in (185, 200, 215, 230):
        _, th = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pads: List[Tuple[float, float, float, int, int, int, int]] = []
        for c in cnts:
            area = float(cv2.contourArea(c))
            # 标记垫：排除内容区大块高光
            if area < 0.0003 * img_area or area > 0.008 * img_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            ar = bw / max(bh, 1)
            if ar < 0.45 or ar > 2.2:
                continue
            if area / float(bw * bh + 1e-6) < 0.35:
                continue
            pads.append((x + bw * 0.5, y + bh * 0.5, area, x, y, bw, bh))
        if len(pads) < 4:
            continue

        pads_sorted = sorted(pads, key=lambda t: t[2], reverse=True)
        for ref in pads_sorted[:6]:
            group = [p for p in pads if 0.55 * ref[2] <= p[2] <= 1.8 * ref[2]]
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

            def inner(pad, which: str) -> np.ndarray:
                _, _, _, x, y, bw, bh = pad
                # 更贴内容内角，减少把标记带进结果
                inset = 0.08
                if which == "tl":
                    return np.array(
                        [x + bw * (1 - inset), y + bh * (1 - inset)], np.float32
                    )
                if which == "tr":
                    return np.array([x + bw * inset, y + bh * (1 - inset)], np.float32)
                if which == "br":
                    return np.array([x + bw * inset, y + bh * inset], np.float32)
                return np.array([x + bw * (1 - inset), y + bh * inset], np.float32)

            quad_s = np.array(
                [
                    inner(p_tl, "tl"),
                    inner(p_tr, "tr"),
                    inner(p_br, "br"),
                    inner(p_bl, "bl"),
                ],
                dtype=np.float32,
            )
            if not _quad_sane(quad_s, h, w, strict=True):
                continue
            geo_s = _quad_geometry_score(quad_s, h, w)
            if geo_s > best_geo:
                best_geo = geo_s
                best_quad = quad_s * inv_scale
                best_meta = {"thr": thr, "n_pads": len(group)}

    if best_quad is None:
        return None
    best_quad = _shrink_quad(best_quad, 0.92)
    if not _quad_sane(best_quad, h0, w0, strict=True):
        return None
    geo = _quad_geometry_score(best_quad, h0, w0)
    conf = estimate_confidence(0, "white_pads", geo, fallback="white_pads")
    # 兜底不再人工抬分；必须靠几何本身过线
    meta = {
        "ok": True,
        "fallback": "white_pads",
        "method": "white_pads",
        "n_required": 0,
        "ids": [],
        "geo": round(geo, 3),
        "confidence": round(conf, 3),
        "fast": True,
        **best_meta,
    }
    return best_quad, meta


def _detect_white_frame_fast(
    bgr: np.ndarray, inv_scale: float, h0: int, w0: int
) -> Optional[Tuple[np.ndarray, dict]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    img_area = float(h * w)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    best = None
    best_geo = -1.0

    for thr_mode in ("otsu", 200, 220):
        if thr_mode == "otsu":
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, th = cv2.threshold(blur, int(thr_mode), 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts[:8]:
            area = cv2.contourArea(c)
            if area < 0.08 * img_area or area > 0.92 * img_area:
                continue
            peri = cv2.arcLength(c, True)
            for eps in (0.02, 0.035, 0.05):
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue
                quad = _order_quad_tl_tr_br_bl(approx)
                if not _quad_sane(quad, h, w, strict=True):
                    continue
                geo = _quad_geometry_score(quad, h, w)
                if geo > best_geo:
                    best_geo = geo
                    best = quad

    if best is None:
        return None
    quad = best * inv_scale
    quad = _shrink_quad(quad, 0.78)  # 白框偏外，多收一点
    if not _quad_sane(quad, h0, w0, strict=True):
        return None
    geo = _quad_geometry_score(quad, h0, w0)
    conf = estimate_confidence(0, "white_frame", geo, fallback="white_frame")
    return quad, {
        "ok": True,
        "fallback": "white_frame",
        "method": "white_frame",
        "n_required": 0,
        "ids": [],
        "geo": round(geo, 3),
        "confidence": round(conf, 3),
        "fast": True,
    }


def detect_content_quad_fast(
    bgr: np.ndarray,
    *,
    max_side: int = 1600,
    min_confidence: float = 0.40,
    try_sharp: bool = True,
    allow_fallback: bool = True,
    allow_white_frame: bool = False,
    min_fallback_confidence: float = 0.58,
    min_aruco_markers: int = 3,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    快速检测内容四边形（坐标相对原图 ``bgr``）。

    - ArUco：缩略图上 raw/clahe/sharp × 两档尺度
    - 默认至少 3 个定位点才接受（2 点补全易畸变）
    - 白垫兜底门槛更高；白框默认关闭（最易畸形）
    - 内容可抖糊；几何不达标 → 失败
    """
    if bgr is None or bgr.size == 0:
        return None, {"ok": False, "reason": "empty", "confidence": 0.0, "fast": True}

    h0, w0 = bgr.shape[:2]
    small, det_scale = _resize_for_detect(bgr, max_side)
    inv = 1.0 / det_scale

    gray0 = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants = [("raw", gray0), ("clahe", clahe.apply(gray0))]
    if try_sharp:
        variants.append(
            (
                "sharp",
                cv2.addWeighted(
                    gray0, 1.55, cv2.GaussianBlur(gray0, (0, 0), 1.2), -0.55, 0
                ),
            )
        )

    candidates: List[Tuple[np.ndarray, dict]] = []
    scales = (1.0, 0.85)

    for vname, gray in variants:
        for sc in scales:
            if abs(sc - 1.0) < 1e-6:
                g = gray
                inv_g = inv
            else:
                g = cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
                inv_g = inv / sc
            for relaxed in (False, True):
                hit = _try_aruco_on_gray(
                    g,
                    inv_g,
                    h0,
                    w0,
                    variant=f"{vname}{'r' if relaxed else ''}",
                    min_markers=2,
                    relaxed=relaxed,
                )
                if hit is None:
                    continue
                q, meta = hit
                meta["det_scale"] = round(det_scale, 3)
                n_req = int(meta["n_required"])
                # 默认丢掉 2 点补全（畸形主因之一）
                if n_req < min_aruco_markers:
                    continue
                if n_req >= 4 and meta["confidence"] >= min_confidence:
                    return q, meta
                candidates.append(hit)
                if n_req >= 3 and meta["confidence"] >= 0.70:
                    break
            else:
                continue
            break
        else:
            continue
        break

    has_solid_aruco = any(
        m.get("n_required", 0) >= 3 and m.get("confidence", 0) >= min_confidence
        for _, m in candidates
    )

    if allow_fallback and not has_solid_aruco:
        pads = _detect_white_pads_fast(small, inv, h0, w0)
        if pads is not None and pads[1].get("confidence", 0) >= min_fallback_confidence:
            candidates.append(pads)
        if allow_white_frame:
            frame = _detect_white_frame_fast(small, inv, h0, w0)
            if (
                frame is not None
                and frame[1].get("confidence", 0) >= max(min_fallback_confidence, 0.65)
            ):
                candidates.append(frame)

    if not candidates:
        return None, {
            "ok": False,
            "reason": "no_markers",
            "confidence": 0.0,
            "fast": True,
        }

    candidates.sort(key=lambda x: _rank(x[1]), reverse=True)
    best_q, best_m = candidates[0]
    method = best_m.get("method", "")
    need = (
        min_fallback_confidence
        if method in ("white_pads", "white_frame")
        else min_confidence
    )
    if best_m.get("confidence", 0.0) >= need and _quad_sane(
        best_q,
        h0,
        w0,
        strict=method in ("white_pads", "white_frame")
        or int(best_m.get("n_required") or 0) < 3,
    ):
        best_m["ok"] = True
        return best_q, best_m

    best_m["ok"] = False
    best_m["reason"] = (
        f"reject method={method} conf={best_m.get('confidence')} "
        f"need>={need} (防畸形)"
    )
    return None, best_m


__all__ = [
    "detect_content_quad_fast",
    "warp_content",
    "draw_debug",
]
