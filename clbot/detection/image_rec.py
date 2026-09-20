import os
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from os.path import abspath, dirname, join

import cv2
import numpy as np

from clbot.utils.image_handler import open_from_path

DEFAULT_TOLERANCE = 0.88
FALLBACK_TOLERANCE = 0.78
BLACK_FRAME_THRESHOLD = 5

# Template cache (Part 7.1): load once at startup, reuse across matches.
_TEMPLATE_CACHE: dict[str, np.ndarray] = {}

# Region-of-interest presets in full-frame coords (Part 7.2).
REGIONS = {
    "elixir_bar": (100, 880, 880, 20),
    "hand_cards": (100, 950, 1720, 130),
    "tower_left": (0, 500, 400, 400),
    "tower_right": (1520, 500, 400, 400),
}


def load_templates(directory: str) -> dict[str, np.ndarray]:
    """Load every .png in directory into the process-wide cache."""
    for fname in os.listdir(directory):
        if fname.endswith(".png"):
            path = join(directory, fname)
            try:
                _TEMPLATE_CACHE[path] = open_from_path(path)
            except Exception:
                continue
    return _TEMPLATE_CACHE


def get_template(name: str) -> np.ndarray | None:
    return _TEMPLATE_CACHE.get(name)


def get_roi(frame: np.ndarray, name: str) -> np.ndarray:
    """Crop a named region; unknown names return the full frame."""
    if name not in REGIONS:
        return frame
    x, y, w, h = REGIONS[name]
    h_frame, w_frame = frame.shape[:2]
    x2, y2 = min(w_frame, x + w), min(h_frame, y + h)
    return frame[y:y2, x:x2]


def is_blank_frame(frame: np.ndarray | None) -> bool:
    """Detect black/blank screenshots that indicate a render failure."""
    if frame is None:
        return True
    try:
        if getattr(frame, "size", 0) == 0:
            return True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(gray.mean()) < BLACK_FRAME_THRESHOLD
    except Exception:
        return True


def save_debug_screenshot(frame: np.ndarray | None, tag: str = "unknown") -> str | None:
    """Persist a failure frame under debug_screens/; best-effort."""
    if frame is None:
        return None
    try:
        os.makedirs("debug_screens", exist_ok=True)
        path = f"debug_screens/{tag}_{int(time.time())}.png"
        cv2.imwrite(path, frame)
        return path
    except Exception:
        return None


def compare_images_multi_scale(
    image: np.ndarray,
    template: np.ndarray,
    threshold: float = DEFAULT_TOLERANCE,
    scales: tuple[float, ...] = (1.0, 0.95, 1.05),
) -> dict | None:
    """Grayscale multi-scale match; returns location/confidence dict or None."""
    best = None
    for scale in scales:
        needle = template if scale == 1.0 else cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        loc = compare_images(image, needle, threshold=threshold)
        if loc is not None:
            return {"location": loc, "confidence": threshold, "scale": scale}
        # Track best-effort confidence for diagnostics.
        try:
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            tpl_gray = cv2.cvtColor(needle, cv2.COLOR_RGB2GRAY)
            if tpl_gray.shape[0] <= img_gray.shape[0] and tpl_gray.shape[1] <= img_gray.shape[1]:
                res = cv2.matchTemplate(img_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if best is None or max_val > best["confidence"]:
                    best = {"location": [int(max_loc[1]), int(max_loc[0])], "confidence": float(max_val), "scale": scale}
        except Exception:
            continue
    return None


def find_image_single(
    haystack: np.ndarray,
    needle_path: str,
    tolerance: float = DEFAULT_TOLERANCE,
    region=None,
    multi_scale: bool = True,
) -> dict | None:
    """Single-template match with fallback tolerances; returns dict or None."""
    if haystack is None or is_blank_frame(haystack):
        return None
    search = haystack
    if region:
        x, y, w, h = region
        search = haystack[y : y + h, x : x + w]
    try:
        needle = get_template(needle_path) or open_from_path(needle_path)
    except Exception:
        return None
    tolerances = [tolerance, (tolerance + FALLBACK_TOLERANCE) / 2, FALLBACK_TOLERANCE]
    scales = (1.0, 0.95, 1.05) if multi_scale else (1.0,)
    best = None
    for scale in scales:
        scaled = needle if scale == 1.0 else cv2.resize(needle, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scaled.shape[0] > search.shape[0] or scaled.shape[1] > search.shape[1]:
            continue
        try:
            img_gray = cv2.cvtColor(search, cv2.COLOR_RGB2GRAY)
            tpl_gray = cv2.cvtColor(scaled, cv2.COLOR_RGB2GRAY)
            res = cv2.matchTemplate(img_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
        except Exception:
            continue
        for tol in tolerances:
            if max_val >= tol:
                return {"location": max_loc, "confidence": float(max_val), "scale": scale, "tolerance_used": tol}
        if best is None or max_val > best["confidence"]:
            best = {"location": max_loc, "confidence": float(max_val), "scale": scale}
    return None

# =============================================================================
# IMAGE RECOGNITION FUNCTIONS
# =============================================================================


def find_image(
    image: np.ndarray,
    folder: str,
    tolerance: float = 0.88,
    subcrop: tuple[int, int, int, int] | None = None,
    show_image: bool = False,
    multi_scale: bool = False,
    fallback_tolerance: float = FALLBACK_TOLERANCE,
) -> tuple[int, int] | None:
    """Find the first matching reference image in a screenshot

    Args:
        image: image to search through
        folder: folder containing reference images (within reference_images directory)
        tolerance: matching tolerance (0.0 to 1.0)
        subcrop: optional subcrop region as (x1, y1, x2, y2) to search within

    Returns:
        tuple[int, int] | None: (x, y) coordinates of found image relative to full image, or None if not found
    """
    if image is None or is_blank_frame(image):
        return None
    search_image = image
    offset_x, offset_y = 0, 0

    if subcrop is not None:
        x1, y1, x2, y2 = subcrop
        search_image = image[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1

    # if show_image:
    #     plt.imshow(search_image)
    #     plt.title(f"Searching for {folder} in image")
    #     plt.show()

    for tol in (tolerance, (tolerance + fallback_tolerance) / 2, fallback_tolerance):
        locations, filenames = find_references(search_image, folder, tol)
        coord = get_first_location(locations)
        if coord is not None:
            # Find which file matched
            for i, location in enumerate(locations):
                if location is not None:
                    print(f"Match found in file: {filenames[i]}")
                    break
            # Convert from [y, x] to (x, y) and add offset to get coordinates relative to full image
            return (coord[1] + offset_x, coord[0] + offset_y)
    return None


def find_references(
    image: np.ndarray,
    folder: str,
    tolerance=0.88,
) -> tuple[list[list[int] | None], list[str]]:
    """Find all reference images in a screenshot

    Args:
    ----
        image (numpy.ndarray): image to find references in
        folder (str): folder to find references (from within reference_images)
        tolerance (float, optional): tolerance. Defaults to 0.88.

    Returns:
    -------
        tuple[list[list[int] | None], list[str]]: coordinate locations and corresponding filenames

    """
    top_level = dirname(__file__)
    reference_folder = abspath(join(top_level, "reference_images", folder))

    filenames = [name for name in os.listdir(reference_folder) if name.endswith(".png") or name.endswith(".jpg")]

    reference_images = [open_from_path(join(reference_folder, name)) for name in filenames]

    with ThreadPoolExecutor(
        max_workers=len(reference_images),
        thread_name_prefix="ImageRecognition",
    ) as executor:
        futures: list[Future[list[int] | None]] = [
            executor.submit(
                compare_images,
                image,
                template,
                tolerance,
            )
            for template in reference_images
        ]
        results = [future.result() for future in as_completed(futures)]
        return results, filenames


def compare_images(
    image: np.ndarray,
    template: np.ndarray,
    threshold=0.8,
):
    """Detects pixel location of a template in an image using template matching

    Args:
        image (numpy.ndarray): image to find template within
        template (numpy.ndarray): template image to match to
        threshold (float, optional): matching threshold. Defaults to 0.8

    Returns:
        list[int] | None: pixel location [y, x] or None if not found
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

    # Check if template is larger than image
    if template_gray.shape[0] > img_gray.shape[0] or template_gray.shape[1] > img_gray.shape[1]:
        return None

    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Get the best match location
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)

    # Return best match if it exceeds threshold
    if max_val >= threshold:
        return [int(max_loc[1]), int(max_loc[0])]  # [y, x]

    return None


# =============================================================================
# PIXEL RECOGNITION FUNCTIONS
# =============================================================================


def pixel_is_equal(
    pix1: tuple[int, int, int] | list[int],
    pix2: tuple[int, int, int] | list[int],
    tol: float,
) -> bool:
    """Check if two pixels are equal within tolerance

    Args:
    ----
        pix1: first RGB pixel
        pix2: second RGB pixel
        tol: color tolerance

    Returns:
    -------
        bool: whether pixels are equal within tolerance

    """
    diff_r = abs(int(pix1[0]) - int(pix2[0]))
    diff_g = abs(int(pix1[1]) - int(pix2[1]))
    diff_b = abs(int(pix1[2]) - int(pix2[2]))
    return (diff_r < tol) and (diff_g < tol) and (diff_b < tol)


def check_line_for_color(
    emulator,
    x_1: int,
    y_1: int,
    x_2: int,
    y_2: int,
    color: tuple[int, int, int],
) -> bool:
    """Check if any pixel along a line matches a specific color

    Args:
        emulator: emulator instance
        x_1, y_1, x_2, y_2: line coordinates
        color: RGB color to check for

    Returns:
        bool: True if any pixel on line matches color
    """
    coordinates = get_line_coordinates(x_1, y_1, x_2, y_2)
    iar = np.asarray(emulator.screenshot())

    for coordinate in coordinates:
        pixel = iar[coordinate[1]][coordinate[0]]
        pixel = convert_pixel(pixel)

        if pixel_is_equal(color, pixel, tol=35):
            return True
    return False


def region_is_color(emulator, region: list, color: tuple[int, int, int]) -> bool:
    """Check if entire region matches a specific color (sampled every 2 pixels)

    Args:
        emulator: emulator instance
        region: [left, top, width, height] region to check
        color: RGB color to check for

    Returns:
        bool: True if entire region matches color
    """
    left, top, width, height = region
    iar = np.asarray(emulator.screenshot())

    for x_index in range(left, left + width, 2):
        for y_index in range(top, top + height, 2):
            pixel = iar[y_index][x_index]
            pixel = convert_pixel(pixel)
            if not pixel_is_equal(color, pixel, tol=35):
                return False

    return True


def all_pixels_are_equal(
    pixels_1: list,
    pixels_2: list,
    tol: float,
) -> bool:
    """Check if two lists of pixels are equal within tolerance

    Args:
        pixels_1: first list of pixels
        pixels_2: second list of pixels
        tol: color tolerance

    Returns:
        bool: True if all pixels match within tolerance
    """
    for pixel1, pixel2 in zip(pixels_1, pixels_2):
        if not pixel_is_equal(pixel1, pixel2, tol):
            return False
    return True


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_first_location(
    locations: list[list[int] | None],
    flip=False,
) -> list[int] | None:
    """Get the first valid location from a list of locations

    Args:
    ----
        locations: list of coordinate locations
        flip: whether to flip x,y coordinates

    Returns:
    -------
        list[int] | None: first valid location or None

    """
    return next(
        ([location[1], location[0]] if flip else location for location in locations if location is not None),
        None,
    )


def check_for_location(locations: list[list[int] | None]) -> bool:
    """Check if any location in the list is valid

    Args:
    ----
        locations: list of coordinate locations

    Returns:
    -------
        bool: True if any location is not None

    """
    return any(location is not None for location in locations)


def convert_pixel(bgr_pixel) -> list[int]:
    """Convert BGR pixel format to RGB

    Args:
        bgr_pixel: pixel in BGR format

    Returns:
        list[int]: pixel in RGB format [red, green, blue]
    """
    red = bgr_pixel[2]
    green = bgr_pixel[1]
    blue = bgr_pixel[0]
    return [red, green, blue]


def get_line_coordinates(x_1: int, y_1: int, x_2: int, y_2: int) -> list[tuple[int, int]]:
    """Get all pixel coordinates along a line using Bresenham's algorithm

    Args:
        x_1, y_1: start coordinates
        x_2, y_2: end coordinates

    Returns:
        list[tuple[int, int]]: list of (x, y) coordinates along the line
    """
    coordinates = []
    delta_x = abs(x_2 - x_1)
    delta_y = abs(y_2 - y_1)
    step_x = -1 if x_1 > x_2 else 1
    step_y = -1 if y_1 > y_2 else 1
    error = delta_x - delta_y

    while x_1 != x_2 or y_1 != y_2:
        coordinates.append((x_1, y_1))
        double_error = 2 * error
        if double_error > -delta_y:
            error -= delta_y
            x_1 += step_x
        if double_error < delta_x:
            error += delta_x
            y_1 += step_y

    coordinates.append((x_1, y_1))
    return coordinates
