"""CHIP-8 rendering utilities for visualization."""

import time
from typing import Tuple, Optional

import numpy as np
import torch


def _to_numpy(display) -> np.ndarray:
    if isinstance(display, torch.Tensor):
        return display.detach().cpu().numpy()
    return np.asarray(display)


def chip8_display_to_rgb(
    display,
    scale: int = 8,
    on_color: Tuple[int, int, int] = (0, 255, 0),
    off_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Convert CHIP-8 boolean display to RGB array with optional upscaling.

    Args:
        display: Boolean array of shape (64, 32) representing CHIP-8 display
        scale: Upscaling factor for better visibility (default: 8x)
        on_color: RGB color for "on" pixels (default: green)
        off_color: RGB color for "off" pixels (default: black)

    Returns:
        RGB array of shape (height*scale, width*scale, 3) with uint8 values
    """
    pixels = _to_numpy(display).astype(np.bool_)

    # Original: (64 width, 32 height) -> Display: (32 height, 64 width)
    pixels = pixels.T
    height, width = pixels.shape

    rgb_frame = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_frame[pixels] = on_color
    rgb_frame[~pixels] = off_color

    if scale > 1:
        rgb_frame = np.repeat(np.repeat(rgb_frame, scale, axis=0), scale, axis=1)
    return rgb_frame


def create_color_scheme(
    scheme: str = "octax",
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Get predefined color schemes for CHIP-8 rendering.

    Args:
        scheme: Color scheme name ("octax", "classic", "amber", "white", "blue", "retro")

    Returns:
        Tuple of (on_color, off_color) as RGB tuples
    """
    schemes = {
        "octax": ((179, 102, 184), (45, 25, 61)),
        "octorch": ((238, 76, 44), (30, 30, 30)),
        "classic": ((0, 255, 0), (0, 0, 0)),
        "amber": ((255, 176, 0), (0, 0, 0)),
        "white": ((255, 255, 255), (0, 0, 0)),
        "blue": ((0, 255, 255), (0, 0, 64)),
        "retro": ((255, 255, 0), (64, 0, 64)),
    }
    if scheme not in schemes:
        raise ValueError(f"Unknown color scheme '{scheme}'. Available: {list(schemes.keys())}")
    return schemes[scheme]


def batch_render(displays, scale: int = 4, color_scheme: str = "octax") -> np.ndarray:
    """Render multiple CHIP-8 displays in a grid layout with transparent spacing.

    Args:
        displays: Array of shape (batch_size, 64, 32) with multiple displays
        scale: Upscaling factor (smaller for batch rendering)
        color_scheme: Color scheme name

    Returns:
        RGBA array showing all displays in a grid layout with transparent padding
    """
    displays = _to_numpy(displays)
    batch_size = displays.shape[0]
    on_color, off_color = create_color_scheme(color_scheme)
    padding = 5

    grid_cols = int(np.ceil(np.sqrt(batch_size)))
    grid_rows = int(np.ceil(batch_size / grid_cols))

    rendered_displays = []
    for i in range(batch_size):
        rgb = chip8_display_to_rgb(displays[i], scale, on_color, off_color)
        rgba = np.concatenate([rgb, 255 * np.ones((*rgb.shape[:2], 1), dtype=np.uint8)], axis=-1)
        rendered_displays.append(rgba)

    while len(rendered_displays) < grid_rows * grid_cols:
        rendered_displays.append(np.zeros_like(rendered_displays[0]))

    display_height, display_width = rendered_displays[0].shape[:2]
    grid_height = grid_rows * display_height + (grid_rows - 1) * padding
    grid_width = grid_cols * display_width + (grid_cols - 1) * padding
    grid_image = np.zeros((grid_height, grid_width, 4), dtype=np.uint8)

    for i, rendered in enumerate(rendered_displays):
        row, col = divmod(i, grid_cols)
        y_start = row * (display_height + padding)
        x_start = col * (display_width + padding)
        grid_image[y_start:y_start + display_height, x_start:x_start + display_width] = rendered
    return grid_image


def create_video(
    displays,
    filename: Optional[str] = None,
    fps: float = 60.0,
    scale: int = 8,
    color_scheme: str = "octax",
    persistence: bool = True,
    display: bool = False,
) -> None:
    """Display and/or save CHIP-8 video with optional phosphor persistence.

    Args:
        displays: EmulatorState (uses ``.display``) or array with shape (N, 64, 32)
        filename: If provided, save video to this MP4 file
        fps: Video frame rate
        scale: Upscaling factor
        color_scheme: Color scheme for rendering
        persistence: Enable phosphor screen simulation (smooth fading)
        display: If True, show video in window (press 'q' to quit, space to pause)
    """
    import cv2  # optional dependency

    if displays is None and display is False:
        return
    if hasattr(displays, "display"):
        displays = displays.display
    displays = _to_numpy(displays)
    if len(displays.shape) != 3 or displays.shape[1:] != (64, 32):
        raise ValueError(f"Expected display shape (N, 64, 32), got {displays.shape}")

    height, width = 32 * scale, 64 * scale
    on_color, off_color = np.array(create_color_scheme(color_scheme))

    writer = None
    if filename:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    if display:
        window_name = "CHIP-8 Video (q=quit, space=pause)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    glow = np.zeros((64, 32), dtype=np.float32) if persistence else None
    decay = 0.8
    frame_delay = 1.0 / fps if display else 0
    paused = False

    try:
        for i, frame_display in enumerate(displays):
            start_time = time.time()
            if persistence:
                glow = np.clip(glow * decay + frame_display.astype(np.float32), 0.0, 1.0)
                pixel_values = glow.T
            else:
                pixel_values = frame_display.T.astype(np.float32)

            frame = np.zeros((32, 64, 3), dtype=np.uint8)
            for c in range(3):
                frame[:, :, c] = off_color[c] + pixel_values * (on_color[c] - off_color[c])
            if scale > 1:
                frame = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if writer:
                writer.write(frame_bgr)

            if display:
                cv2.putText(frame_bgr, f"Frame {i + 1}/{len(displays)}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(window_name, frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord(" "):
                    paused = not paused
                    if paused:
                        print("Paused - press space to continue, 'q' to quit")
                while paused:
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord(" "):
                        paused = False
                        print("Resumed")
                        break
                    elif key == ord("q") or key == 27:
                        return
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_delay - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    if filename:
        duration = len(displays) / fps
        print(f"Video saved: {filename} ({len(displays)} frames, {fps} FPS, {duration:.1f}s)")
    if display:
        print("Display closed")
