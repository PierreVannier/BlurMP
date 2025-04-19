#!/usr/bin/env python3
"""Blurry Vision MP - Parallel processing version using Tesseract OCR."""

import argparse
import math
import multiprocessing
import re
import shutil
import subprocess
import time
import unicodedata
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from rich import print as rich_print

# Attempt to import pytesseract for optional OCR fallback
try:
    import pytesseract
except ImportError:
    logger.critical(
        "Required library pytesseract not found. Please install it and Tesseract OCR: pip install pytesseract"
    )
    pytesseract = None  # Will be checked at runtime


# --------------------------------------------------------------------------- #
#  Utility helpers
# --------------------------------------------------------------------------- #
def blur_region(frame: np.ndarray, bbox: list, kernel_size: int = 51) -> np.ndarray:
    """Applies Gaussian blur to a specified bounding box region in a frame.

    The kernel size is automatically reduced if it is larger than the region
    being blurred so that OpenCV does not raise an error for small ROIs.
    """
    try:
        if len(bbox) != 4:
            logger.warning(
                f"Invalid bounding box point count: {len(bbox)}, expected 4. Box: {bbox}"
            )
            return frame

        tl_x, tl_y = map(int, bbox[0])
        br_x, br_y = map(int, bbox[2])

        h, w = frame.shape[:2]
        x_min, y_min = max(0, tl_x), max(0, tl_y)
        x_max, y_max = min(w, br_x), min(h, br_y)

        if x_min >= x_max or y_min >= y_max:
            return frame  # invalid box

        roi = frame[y_min:y_max, x_min:x_max]
        roi_h, roi_w = roi.shape[:2]
        if roi_h == 0 or roi_w == 0:
            return frame  # nothing to blur

        # Ensure kernel size is odd and does not exceed ROI dimensions
        max_kernel = max(3, min(roi_w, roi_h))
        effective_kernel = min(kernel_size, max_kernel)
        if effective_kernel % 2 == 0:
            effective_kernel -= 1  # make it odd
        if effective_kernel < 3:
            return frame  # kernel too small, skip

        blurred_roi = cv2.GaussianBlur(roi, (effective_kernel, effective_kernel), 0)
        frame[y_min:y_max, x_min:x_max] = blurred_roi
    except Exception as e:
        logger.error(f"Error blurring region with bbox {bbox}: {e}", exc_info=True)
    return frame


# --------------------------------------------------------------------------- #
#  Pixelation helper
# --------------------------------------------------------------------------- #
def pixelate_region(frame: np.ndarray, bbox: list, pixel_size: int = 16) -> np.ndarray:
    """Applies pixelation to a specified bounding box region in a frame."""
    try:
        if len(bbox) != 4:
            logger.warning(
                f"Invalid bounding box point count: {len(bbox)}, expected 4. Box: {bbox}"
            )
            return frame

        tl_x, tl_y = map(int, bbox[0])
        br_x, br_y = map(int, bbox[2])

        h, w = frame.shape[:2]
        x_min, y_min = max(0, tl_x), max(0, tl_y)
        x_max, y_max = min(w, br_x), min(h, br_y)

        if x_min >= x_max or y_min >= y_max:
            return frame  # invalid box

        roi = frame[y_min:y_max, x_min:x_max]
        roi_h, roi_w = roi.shape[:2]
        if roi_h == 0 or roi_w == 0:
            return frame  # nothing to pixelate

        # Ensure pixel size is positive
        eff_pixel_size = max(1, pixel_size)

        # Resize down to pixelated dimensions
        temp = cv2.resize(
            roi,
            (max(1, roi_w // eff_pixel_size), max(1, roi_h // eff_pixel_size)),
            interpolation=cv2.INTER_LINEAR,
        )

        # Resize back up to original ROI size using nearest neighbor
        pixelated_roi = cv2.resize(
            temp, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST
        )

        frame[y_min:y_max, x_min:x_max] = pixelated_roi
    except Exception as e:
        logger.error(f"Error pixelating region with bbox {bbox}: {e}", exc_info=True)
    return frame


# --------------------------------------------------------------------------- #
#  Fill helper
# --------------------------------------------------------------------------- #
def fill_region(
    frame: np.ndarray, bbox: list, color: tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """Fills a specified bounding box region with a solid color."""
    try:
        if len(bbox) != 4:
            logger.warning(
                f"Invalid bounding box point count: {len(bbox)}, expected 4. Box: {bbox}"
            )
            return frame

        tl_x, tl_y = map(int, bbox[0])
        br_x, br_y = map(int, bbox[2])

        h, w = frame.shape[:2]
        x_min, y_min = max(0, tl_x), max(0, tl_y)
        x_max, y_max = min(w, br_x), min(h, br_y)

        if x_min >= x_max or y_min >= y_max:
            return frame  # invalid box

        cv2.rectangle(
            frame, (x_min, y_min), (x_max, y_max), color, thickness=-1
        )  # -1 thickness fills the rectangle

    except Exception as e:
        logger.error(f"Error filling region with bbox {bbox}: {e}", exc_info=True)
    return frame


# --------------------------------------------------------------------------- #
#  Processing loop for a single chunk (to be run in parallel)
# --------------------------------------------------------------------------- #
def process_video(
    chunk_idx: int,
    start_frame: int,
    end_frame: int,
    src: Path,
    dst: Path,
    terms: list[str],
    segment: int = 15,
    kernel: int = 51,
    draw_only: bool = False,
    x_offset: int = 0,
    y_offset: int = 0,
    w_scale: float = 1.0,
    h_scale: float = 1.0,
    time_ranges: list[tuple[float, float]] | None = None,
    backtrack_frames: int = 0,
    blur_type: str = "gaussian",
    pixel_size: int = 16,
    fill_color: tuple[int, int, int] = (0, 0, 0),
):
    """Process a specific frame range (chunk) of the video."""
    # Only OCR is supported now
    if pytesseract is None:
        logger.error(
            f"[Chunk {chunk_idx}] Pytesseract OCR library is required but not found."
        )
        return False

    if not src.is_file():
        logger.error(f"[Chunk {chunk_idx}] Input video not found: {src}")
        return False

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        logger.error(f"[Chunk {chunk_idx}] Could not open {src}")
        return False

    W, H = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

    dst.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(dst), fourcc, fps, (W, H))
    if not out.isOpened():
        logger.error(f"[Chunk {chunk_idx}] Failed to create output file {dst}")
        cap.release()
        return False

    logger.info(
        f"[Chunk {chunk_idx}] Processing frames {start_frame}-{end_frame} of {src.name} -> {dst.name}"
    )

    # Move capture object to start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    segment_boxes = {}
    frame_idx = start_frame
    tic = time.time()

    # Convert time_ranges (seconds) to frame indices once we know fps
    if time_ranges:
        ranges_frames = [(int(s * fps), int(e * fps)) for s, e in time_ranges]
    else:
        ranges_frames = None

    # ------ Tracker helper ------ #
    def _create_tracker():
        """Return the best available OpenCV tracker instance or None."""
        legacy = getattr(cv2, "legacy", None)

        # Helper
        def _make(ns, name):
            return getattr(ns, name)() if ns and hasattr(ns, name) else None

        for creator in (
            (cv2, "TrackerCSRT_create"),
            (legacy, "TrackerCSRT_create"),
            (cv2, "TrackerKCF_create"),
            (legacy, "TrackerKCF_create"),
            (cv2, "TrackerMOSSE_create"),
            (legacy, "TrackerMOSSE_create"),
        ):
            ns, fn = creator
            obj = _make(ns, fn)
            if obj is not None:
                return obj
        return None

    active_trackers: list[cv2.Tracker] = []

    # --------------------------------------------------------------
    #  Backtrack frame buffer (keeps recent frames before encoding)
    # --------------------------------------------------------------
    frame_buffer: deque[np.ndarray] | None = (
        deque(maxlen=backtrack_frames + 1) if backtrack_frames > 0 else None
    )

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            logger.warning(
                f"[Chunk {chunk_idx}] Could not read frame {frame_idx}, stopping chunk early."
            )
            break

        # Check if the frame is OUTSIDE the desired time ranges
        is_in_range = True
        if ranges_frames and not any(
            start <= frame_idx <= end for start, end in ranges_frames
        ):
            is_in_range = False

        # If outside the range, write directly and skip blurring/buffering
        if not is_in_range:
            out.write(frame)
            frame_idx += 1
            continue

        # ----------------------------------------------------------
        #  Store an unmodified copy in the buffer for retro‑blur
        # ----------------------------------------------------------
        if frame_buffer is not None:
            frame_buffer.append(frame.copy())

        seg = frame_idx // segment
        if seg not in segment_boxes:  # Run detection once per segment
            logger.info(
                f"[Chunk {chunk_idx}] --- Starting analysis for segment {seg} (Frame {frame_idx}) ---"
            )
            all_bboxes_for_segment = []

            # -------- OCR: run once and match all targets -------- #
            try:
                # run pytesseract once
                h_full, w_full = frame.shape[:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                config = "--oem 3 --psm 6"
                data = pytesseract.image_to_data(
                    gray, output_type=pytesseract.Output.DICT, config=config
                )

                def _norm(t: str):
                    t = t.lower()
                    t = "".join(
                        c
                        for c in unicodedata.normalize("NFKD", t)
                        if not unicodedata.combining(c)
                    )
                    return re.sub(r"\W+", "", t)

                words_norm = [_norm(t) for t in data["text"]]
                n_words = len(words_norm)

                for target_text in terms:
                    target_tokens = [_norm(tok) for tok in target_text.split()]
                    if not target_tokens:
                        continue
                    i = 0
                    while i < n_words:
                        if words_norm[i] == target_tokens[0]:
                            match = True
                            for j in range(1, len(target_tokens)):
                                if (
                                    i + j >= n_words
                                    or words_norm[i + j] != target_tokens[j]
                                ):
                                    match = False
                                    break
                            if match:
                                xs, ys = [], []
                                for j in range(len(target_tokens)):
                                    xs.extend(
                                        [
                                            data["left"][i + j],
                                            data["left"][i + j] + data["width"][i + j],
                                        ]
                                    )
                                    ys.extend(
                                        [
                                            data["top"][i + j],
                                            data["top"][i + j] + data["height"][i + j],
                                        ]
                                    )
                                x_min, x_max = max(0, min(xs)), min(w_full, max(xs))
                                y_min, y_max = max(0, min(ys)), min(h_full, max(ys))
                                if x_min < x_max and y_min < y_max:
                                    bbox = [
                                        [x_min, y_min],
                                        [x_max, y_min],
                                        [x_max, y_max],
                                        [x_min, y_max],
                                    ]
                                    all_bboxes_for_segment.append(bbox)
                                i += len(target_tokens)
                                continue
                        i += 1
            except Exception as ocr_e:
                logger.error(f"[Chunk {chunk_idx}] Optimised OCR failed: {ocr_e}")

            segment_boxes[seg] = all_bboxes_for_segment
            logger.info(
                f"[Chunk {chunk_idx}] Segment {seg}: Found {len(all_bboxes_for_segment)} regions via OCR."
            )

            # (Re)initialise trackers for this segment
            active_trackers.clear()
            for bbox in all_bboxes_for_segment:
                x1, y1 = bbox[0]
                x2, y2 = bbox[2]
                trk = _create_tracker()
                if trk is None:
                    break  # trackers unavailable
                # Use the frame being analyzed for detection for tracker init
                trk.init(frame, (x1, y1, x2 - x1, y2 - y1))
                active_trackers.append(trk)

        # Update trackers to follow motion
        tracked_boxes: list[list] = []
        if seg in segment_boxes:
            for trk in list(active_trackers):
                # Update tracker on the current frame
                ok, box = trk.update(frame)
                if not ok:
                    active_trackers.remove(trk)
                    continue
                x, y, w_box, h_box = box
                x1, y1, x2, y2 = int(x), int(y), int(x + w_box), int(y + h_box)
                if x1 < x2 and y1 < y2:
                    tracked_boxes.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

        # Determine which boxes to use for the current frame
        if seg in segment_boxes and frame_idx > start_frame + (
            seg * segment
        ):  # Use tracked boxes only after the first frame of the segment
            current_boxes = tracked_boxes
        else:
            # First frame of the segment (or fallback)
            current_boxes = segment_boxes.get(seg, [])

        # Apply adjustments and blur/draw
        adjusted_bboxes: list[list] = []  # Initialize here, outside the condition
        frame_to_write = frame  # Start with the original frame read for this iteration

        if current_boxes:
            for raw_bbox_points in current_boxes:
                adjusted_bbox = raw_bbox_points  # Default to original
                try:
                    # --- Adjustment Calculation ---
                    xs = [p[0] for p in raw_bbox_points]
                    ys = [p[1] for p in raw_bbox_points]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    center_x = (min_x + max_x) / 2.0
                    center_y = (min_y + max_y) / 2.0
                    width = max_x - min_x
                    height = max_y - min_y

                    adj_center_x = center_x + x_offset
                    adj_center_y = center_y + y_offset
                    adj_width = width * w_scale
                    adj_height = height * h_scale

                    adj_tl_x = int(round(adj_center_x - adj_width / 2.0))
                    adj_tl_y = int(round(adj_center_y - adj_height / 2.0))
                    adj_br_x = int(round(adj_center_x + adj_width / 2.0))
                    adj_br_y = int(round(adj_center_y + adj_height / 2.0))

                    h_frame, w_frame = frame.shape[:2]
                    adj_tl_x = max(0, adj_tl_x)
                    adj_tl_y = max(0, adj_tl_y)
                    adj_br_x = min(w_frame, adj_br_x)
                    adj_br_y = min(h_frame, adj_br_y)

                    if adj_tl_x < adj_br_x and adj_tl_y < adj_br_y:
                        adjusted_bbox = [
                            [adj_tl_x, adj_tl_y],
                            [adj_br_x, adj_tl_y],
                            [adj_br_x, adj_br_y],
                            [adj_tl_x, adj_br_y],
                        ]
                    else:
                        logger.warning(
                            f"[Chunk {chunk_idx}] Adjusted bbox invalid: tl=({adj_tl_x},{adj_tl_y}), br=({adj_br_x},{adj_br_y}). Using original."
                        )
                        adjusted_bbox = raw_bbox_points  # Fallback
                except Exception as adj_err:
                    logger.error(
                        f"[Chunk {chunk_idx}] Error adjusting bbox {raw_bbox_points}: {adj_err}"
                    )
                    adjusted_bbox = raw_bbox_points  # Fallback

                # --- Use adjusted_bbox for drawing or blurring on *current* frame --- #
                # Apply effect to the frame_to_write variable regardless of buffering
                if draw_only:
                    try:
                        if len(adjusted_bbox) == 4 and all(
                            len(p) == 2 for p in adjusted_bbox
                        ):
                            pts = np.array(adjusted_bbox, np.int32).reshape((-1, 1, 2))
                            cv2.polylines(
                                frame_to_write,
                                [pts],
                                isClosed=True,
                                color=(0, 0, 255),
                                thickness=2,
                            )  # Red box
                        else:
                            logger.warning(
                                f"[Chunk {chunk_idx}] Skipping drawing invalid adjusted bbox structure: {adjusted_bbox}"
                            )
                    except Exception as draw_err:
                        logger.error(
                            f"[Chunk {chunk_idx}] Error drawing adjusted bbox {adjusted_bbox}: {draw_err}"
                        )
                else:
                    if blur_type == "fill":
                        frame_to_write = fill_region(
                            frame_to_write, adjusted_bbox, color=fill_color
                        )
                    elif blur_type == "pixelate":
                        frame_to_write = pixelate_region(
                            frame_to_write, adjusted_bbox, pixel_size=pixel_size
                        )
                    else:  # Default to gaussian
                        frame_to_write = blur_region(
                            frame_to_write, adjusted_bbox, kernel_size=kernel
                        )

                # Remember bbox for retro‑application
                adjusted_bboxes.append(adjusted_bbox)

            # --- Apply treatment (only if boxes were found/tracked in *this* cycle) --- #
            frame_to_write = frame_to_write  # Start with the original frame read
            if (
                adjusted_bboxes
            ):  # Apply effects if boxes exist for the current frame_idx
                if frame_buffer is None:  # No backtracking -> modify frame directly
                    if draw_only:
                        # Draw on frame (only if not buffering)
                        try:
                            if len(adjusted_bbox) == 4 and all(
                                len(p) == 2 for p in adjusted_bbox
                            ):
                                pts = np.array(adjusted_bbox, np.int32).reshape(
                                    (-1, 1, 2)
                                )
                                cv2.polylines(
                                    frame_to_write,
                                    [pts],
                                    isClosed=True,
                                    color=(0, 0, 255),
                                    thickness=2,
                                )  # Red box
                            else:
                                logger.warning(
                                    f"[Chunk {chunk_idx}] Skipping drawing invalid adjusted bbox structure: {adjusted_bbox}"
                                )
                        except Exception as draw_err:
                            logger.error(
                                f"[Chunk {chunk_idx}] Error drawing adjusted bbox {adjusted_bbox}: {draw_err}"
                            )
                    else:  # Default to gaussian
                        # Blur frame (only if not buffering)
                        frame_to_write = blur_region(
                            frame_to_write, adjusted_bbox, kernel_size=kernel
                        )
                else:
                    # Effects will be applied to the buffer copies, not the 'frame' variable
                    pass

        # Retroactively apply the *same* effects to buffered frames if needed
        if frame_buffer is not None and adjusted_bboxes:
            # Apply retroactively to ALL frames currently in the buffer
            # This includes the frame just added
            for idx in range(len(frame_buffer)):
                for bb in adjusted_bboxes:
                    if draw_only:
                        try:
                            pts = np.array(bb, np.int32).reshape((-1, 1, 2))
                            cv2.polylines(
                                frame_buffer[idx],
                                [pts],
                                isClosed=True,
                                color=(0, 0, 255),
                                thickness=2,
                            )
                        except Exception:
                            pass
                    else:
                        if blur_type == "fill":
                            frame_buffer[idx] = fill_region(
                                frame_buffer[idx], bb, color=fill_color
                            )
                        elif blur_type == "pixelate":
                            frame_buffer[idx] = pixelate_region(
                                frame_buffer[idx], bb, pixel_size=pixel_size
                            )
                        else:  # Default to gaussian
                            frame_buffer[idx] = blur_region(
                                frame_buffer[idx], bb, kernel_size=kernel
                            )

        # ----------------------------------------------------------
        #  Emit the oldest frame if buffer length exceeded
        # ----------------------------------------------------------
        if frame_buffer is not None:
            if len(frame_buffer) > backtrack_frames:
                out.write(
                    frame_buffer.popleft()
                )  # Write the oldest, now modified frame
        else:
            # If no backtracking, write the processed frame directly
            out.write(frame_to_write)

        frame_idx += 1

        # --- Progress specific to chunk --- #
        frames_in_chunk = end_frame - start_frame
        processed_in_chunk = frame_idx - start_frame
        chunk_elapsed = time.time() - tic
        chunk_eta = (
            (chunk_elapsed / processed_in_chunk)
            * (frames_in_chunk - processed_in_chunk)
            if processed_in_chunk > 0
            else 0
        )
        logger.info(
            f"[Chunk {chunk_idx}] Frame {frame_idx}/{end_frame - 1} ({processed_in_chunk}/{frames_in_chunk}) | Elapsed: {chunk_elapsed:.1f}s | ETA: {chunk_eta:.1f}s"
        )

    cap.release()
    out.release()

    # Flush remaining buffered frames
    if frame_buffer is not None:
        while frame_buffer:
            out.write(frame_buffer.popleft())

    logger.success(
        f"[Chunk {chunk_idx}] Done ✔  Output saved to {dst}  "
        f"({frame_idx - start_frame} frames, {(time.time() - tic):.1f}s)"
    )
    return True


# --------------------------------------------------------------------------- #
#  CLI and Multiprocessing Orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blur/outline text in video using parallel processing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_video", type=Path)
    parser.add_argument("output_video", type=Path)
    parser.add_argument(
        "-t",
        "--texts",
        action="append",
        default=[],
        required=False,
        help="Target string (use multiple -t for several).",
    )
    parser.add_argument(
        "-f",
        "--strings-file",
        type=Path,
        required=False,
        help="Text file with one target string per line.",
    )
    parser.add_argument(
        "-sf",
        "--segment-frames",
        type=int,
        default=15,
        help="Analyze 1 frame every N frames (≥1). Lower means more reactive.",
    )
    parser.add_argument("-k", "--kernel-size", type=int, default=51)
    parser.add_argument(
        "--draw-boxes",
        action="store_true",
        help="Draw red rectangles instead of blurring (debug).",
    )
    parser.add_argument(
        "--x-offset",
        type=int,
        default=0,
        help="Pixels to shift bounding box horizontally (+right, -left).",
    )
    parser.add_argument(
        "--y-offset",
        type=int,
        default=0,
        help="Pixels to shift bounding box vertically (+down, -up).",
    )
    parser.add_argument(
        "--w-scale",
        type=float,
        default=1.0,
        help="Factor to scale bounding box width (e.g., 1.1 is 10%% wider).",
    )
    parser.add_argument(
        "--h-scale",
        type=float,
        default=1.0,
        help="Factor to scale bounding box height (e.g., 0.9 is 10%% shorter).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--time-range",
        action="append",
        default=[],
        metavar="START-END",
        help="Process only this time range (HH:MM:SS-HH:MM:SS). Can be repeated.",
    )
    parser.add_argument(
        "--backtrack",
        type=int,
        default=0,
        metavar="N",
        help="Blur the same region N previous frames retroactively (0 disables).",
    )
    parser.add_argument(
        "--blur-type",
        choices=["gaussian", "pixelate", "fill"],
        default="gaussian",
        help="Type of blur effect to apply.",
    )
    parser.add_argument(
        "--pixel-size",
        type=int,
        default=16,
        help="Size of blocks for pixelation blur (larger is coarser).",
    )
    parser.add_argument(
        "--fill-color",
        type=str,
        default="0,0,0",
        help="BGR color for fill blur (e.g., '0,0,0' for black, '255,255,255' for white).",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel processes to use (defaults to CPU core count).",
    )
    args = parser.parse_args()

    # aggregate target terms
    targets = set(args.texts or [])
    if args.strings_file and args.strings_file.is_file():
        targets.update(
            ln.strip()
            for ln in args.strings_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        )
    if not targets:
        parser.error("No target text provided (use -t or -f).")

    rich_print("[bold blue]Starting Blurry Vision MP (OCR Edition)...[/bold blue]")
    rich_print(f"[cyan]Using {args.jobs} parallel jobs[/cyan]")
    logger.info(f"Input Video: {args.input_video}")
    logger.info(f"Output Video: {args.output_video}")
    logger.info(f"Texts to Blur: {sorted(targets)}")
    logger.info(f"Segment Size: {args.segment_frames} frames")
    logger.info(f"Blur Kernel Size: {args.kernel_size}")
    logger.info(
        f"X Offset: {args.x_offset}, Y Offset: {args.y_offset}, W Scale: {args.w_scale}, H Scale: {args.h_scale}"
    )
    if args.draw_boxes:
        logger.info("Draw Boxes Mode Enabled (Debugging)")

    # --- Validate Tesseract dependency ---
    if pytesseract is None:
        logger.critical("Cannot proceed without pytesseract library.")
        return

    def _parse_hms(ts: str) -> float:
        parts = [int(p) for p in ts.split(":")]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m, s = 0, parts[0], parts[1]
        else:
            h, m, s = 0, 0, parts[0]
        return h * 3600 + m * 60 + s

    time_ranges_sec: list[tuple[float, float]] = []
    for rng in args.time_range:
        try:
            start_str, end_str = rng.split("-")
            time_ranges_sec.append((_parse_hms(start_str), _parse_hms(end_str)))
        except Exception:
            parser.error(f"Invalid --time-range format: {rng}")

    # --- Get Video Properties --- #
    cap = cv2.VideoCapture(str(args.input_video))
    if not cap.isOpened():
        logger.critical(f"Cannot open input video: {args.input_video}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if total_frames <= 0 or fps <= 0:
        logger.critical("Could not read video properties (total frames, fps).")
        return

    logger.info(
        f"Video properties: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames"
    )

    # --- Prepare Chunks --- #
    num_jobs = max(1, args.jobs)
    frames_per_job = math.ceil(total_frames / num_jobs)
    chunks = []
    temp_files = []
    temp_dir = args.output_video.parent / f"_temp_{args.output_video.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_jobs):
        start_frame = i * frames_per_job
        end_frame = min((i + 1) * frames_per_job, total_frames)
        if start_frame >= total_frames:
            continue

        temp_output_path = temp_dir / f"chunk_{i:03d}.mp4"
        temp_files.append(temp_output_path)

        try:
            fill_color_tuple = tuple(map(int, args.fill_color.split(",")))
            if len(fill_color_tuple) != 3:
                raise ValueError("Fill color must have 3 components")
        except ValueError:
            parser.error(
                f"Invalid --fill-color format: '{args.fill_color}'. Use B,G,R format like '0,0,0'."
            )

        chunk_args = (
            i,  # chunk_idx
            start_frame,
            end_frame,
            args.input_video,
            temp_output_path,
            sorted(targets),
            max(args.segment_frames, 1),
            max(3, args.kernel_size | 1),
            args.draw_boxes,
            args.x_offset,
            args.y_offset,
            args.w_scale,
            args.h_scale,
            time_ranges_sec or None,
            args.backtrack,
            args.blur_type,
            args.pixel_size,
            fill_color_tuple,
        )
        chunks.append(chunk_args)

    # --- Run in Parallel --- #
    start_time_total = time.time()
    logger.info(f"Starting parallel processing of {len(chunks)} chunks...")
    with multiprocessing.Pool(processes=num_jobs) as pool:
        results = pool.starmap(process_video, chunks)

    if not all(results):
        logger.critical("One or more chunks failed to process. Aborting merge.")
        # Optionally add cleanup of temp files here
        # shutil.rmtree(temp_dir)
        return

    logger.success(f"All {len(chunks)} chunks processed successfully.")

    # --- Concatenate Chunks using ffmpeg --- #
    logger.info("Concatenating video chunks using ffmpeg...")
    concat_list_path = temp_dir / "concat_list.txt"
    with open(concat_list_path, "w") as f:
        for temp_file in temp_files:
            # Write only the filename relative to the list file location
            f.write(f"file '{temp_file.name}'\n")

    ffmpeg_cmd = [
        "ffmpeg",
        "-f",
        "concat",
        "-safe",
        "0",  # Allow unsafe paths if needed, adjust if causes issues
        "-i",
        str(concat_list_path),
        "-c",
        "copy",  # Copy streams directly without re-encoding
        "-movflags",
        "+faststart",
        "-y",  # Overwrite output if exists
        str(args.output_video),
    ]

    try:
        logger.debug(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
        # Using check=True will raise CalledProcessError if ffmpeg fails
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        logger.debug(f"ffmpeg stdout:\n{result.stdout}")
        logger.debug(f"ffmpeg stderr:\n{result.stderr}")
        logger.success(f"Final video saved to {args.output_video}")
    except FileNotFoundError:
        logger.critical(
            "ffmpeg command not found. Please install ffmpeg and ensure it's in your PATH."
        )
        return
    except subprocess.CalledProcessError as e:
        logger.critical(f"ffmpeg concatenation failed (return code {e.returncode}):")
        logger.error(f"ffmpeg stdout:\n{e.stdout}")
        logger.error(f"ffmpeg stderr:\n{e.stderr}")
        return
    finally:
        # --- Cleanup --- #
        logger.info(f"Cleaning up temporary directory: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
            logger.debug("Temporary directory removed.")
        except Exception as cleanup_err:
            logger.warning(
                f"Failed to remove temporary directory {temp_dir}: {cleanup_err}"
            )

    end_time_total = time.time()
    logger.success(
        f"Total processing time: {end_time_total - start_time_total:.2f} seconds"
    )


if __name__ == "__main__":
    # Freeze support is necessary for multiprocessing on some platforms (like Windows)
    multiprocessing.freeze_support()
    main()
