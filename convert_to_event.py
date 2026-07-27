"""
Enhanced Event Camera (DVS) Simulation
=======================================
Converts standard RGB images into realistic DVS-like event frames using:
  1. Log-intensity temporal contrast (simulates brightness change detection)
  2. Multi-scale Laplacian of Gaussian edge detection
  3. Positive/negative polarity events (ON=red, OFF=blue)
  4. Gaussian noise to simulate sensor noise
  5. Optional 3-channel BGR output for model compatibility

Outputs:
  - reference_map_event.png
  - test_frame_event.png
"""

import cv2
import numpy as np
import os

BASE = "/home/nalin/gps_denied_project"


def simulate_dvs_events(image_path, output_path, contrast_threshold=0.15, noise_std=0.02):
    """
    Simulate DVS event camera output from a single static image.

    Real event cameras detect per-pixel log-intensity changes over time.
    For a static image, we simulate this by:
      1. Computing log-intensity gradients (spatial proxy for temporal contrast)
      2. Multi-scale edge detection via Laplacian of Gaussian
      3. Splitting into positive (ON) and negative (OFF) polarity events
      4. Adding realistic sensor noise

    Args:
        image_path: Path to input RGB image.
        output_path: Path to save the event-frame output.
        contrast_threshold: Minimum log-intensity change to trigger an event (lower = more events).
        noise_std: Standard deviation of Gaussian noise added to simulate sensor noise.

    Returns:
        event_bgr: The 3-channel event visualization (BGR).
    """
    print(f"  Loading {image_path} ...")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    # Convert to grayscale float and compute log-intensity (like a real DVS sensor)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    # Avoid log(0) by clamping minimum intensity
    gray = np.clip(gray, 1.0, 255.0)
    log_intensity = np.log(gray)

    # --- Multi-scale Laplacian of Gaussian (LoG) for edge/event detection ---
    # Real DVS events fire at edges and texture boundaries; LoG captures this
    # at multiple spatial scales to simulate different temporal frequencies.
    scales = [1.0, 2.0, 4.0]
    log_response = np.zeros_like(log_intensity)
    for sigma in scales:
        ksize = int(sigma * 6) | 1  # ensure odd kernel size
        blurred = cv2.GaussianBlur(log_intensity, (ksize, ksize), sigma)
        lap = cv2.Laplacian(blurred, cv2.CV_64F)
        log_response += np.abs(lap) / len(scales)

    # --- Spatial gradient magnitude as temporal contrast proxy ---
    # Scharr gives better rotational symmetry than Sobel
    grad_x = cv2.Scharr(log_intensity, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(log_intensity, cv2.CV_64F, 0, 1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Combine LoG and gradient responses
    combined = 0.6 * grad_mag + 0.4 * log_response

    # Add Gaussian noise to simulate DVS sensor noise
    noise = np.random.normal(0, noise_std, combined.shape)
    combined += noise

    # --- Generate polarity events ---
    # Positive events (ON): brightness increase -> shown as RED
    # Negative events (OFF): brightness decrease -> shown as BLUE
    # Use the sign of the Laplacian to determine polarity
    lap_sign = cv2.Laplacian(log_intensity, cv2.CV_64F)

    on_events = (combined > contrast_threshold) & (lap_sign > 0)
    off_events = (combined > contrast_threshold) & (lap_sign <= 0)

    # Compute event intensity (how strongly the event fires)
    event_strength = np.clip(combined / (contrast_threshold * 5), 0, 1)

    # --- Create 3-channel event visualization ---
    # Black background, red = ON events, blue = OFF events
    event_bgr = np.zeros((*gray.shape, 3), dtype=np.uint8)

    # ON events -> Red channel (BGR: channel 2)
    event_bgr[on_events, 2] = (event_strength[on_events] * 255).astype(np.uint8)
    # OFF events -> Blue channel (BGR: channel 0)
    event_bgr[off_events, 0] = (event_strength[off_events] * 255).astype(np.uint8)

    # Add a subtle green channel for both to give slight white tint at high intensity
    both = on_events | off_events
    event_bgr[both, 1] = (event_strength[both] * 40).astype(np.uint8)

    cv2.imwrite(output_path, event_bgr)
    n_on = on_events.sum()
    n_off = off_events.sum()
    total_pixels = gray.shape[0] * gray.shape[1]
    print(f"  Events: {n_on + n_off:,} total ({n_on:,} ON, {n_off:,} OFF) out of {total_pixels:,} pixels")
    print(f"  Event density: {(n_on + n_off) / total_pixels * 100:.1f}%")
    print(f"  Saved -> {output_path}")
    return event_bgr


if __name__ == "__main__":
    print("=" * 70)
    print("  DVS Event Camera Simulation")
    print("=" * 70)

    ref_path = os.path.join(BASE, "reference_map.png")
    test_path = os.path.join(BASE, "test_frame.png")

    ref_out = os.path.join(BASE, "reference_map_event.png")
    test_out = os.path.join(BASE, "test_frame_event.png")

    print("\n  Converting Reference Map ...")
    simulate_dvs_events(ref_path, ref_out)

    print("\n  Converting Test Frame ...")
    simulate_dvs_events(test_path, test_out)

    print("\n" + "=" * 70)
    print("  Done! Event frames saved.")
    print("=" * 70)
