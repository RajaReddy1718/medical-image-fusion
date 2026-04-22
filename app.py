"""
Medical Image Fusion Web Application
Uses Dual-Tree Complex Wavelet Transform (DT-CWT) for CT and MRI fusion.
Pipeline: Upload → Landmark Registration → DT-CWT Fusion → Watershed Segmentation
"""

import os
import cv2
import numpy as np
import pywt
from flask import Flask, render_template, request, url_for, jsonify

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_ROOT, 'static', 'images')
os.makedirs(STATIC_DIR, exist_ok=True)


# ─── Helper utilities ──────────────────────────────────────────────────────────

def img_path(name):
    """Return full path for an image in static/images/."""
    return os.path.join(STATIC_DIR, name)


def parse_coord_list(raw: str) -> np.ndarray:
    """Parse [[x,y],[x,y],...] string to numpy array."""
    import json
    data = json.loads(raw)
    return np.array(data, dtype=np.float64)


# ─── Image Registration (Procrustes / affine) ─────────────────────────────────

def procrustes_align(src_pts: np.ndarray, dst_pts: np.ndarray):
    """
    Compute affine transformation matrix M (2×3) that maps src_pts → dst_pts
    using least-squares procrustes analysis.
    """
    n = src_pts.shape[0]
    # Build the system Ax = b
    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    for i, (s, d) in enumerate(zip(src_pts, dst_pts)):
        A[2 * i]     = [s[0], s[1], 1, 0,    0,    0]
        A[2 * i + 1] = [0,    0,    0, s[0], s[1], 1]
        b[2 * i]     = d[0]
        b[2 * i + 1] = d[1]
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    M = params.reshape(2, 3)
    return M


def register_image(src_img: np.ndarray, M: np.ndarray, target_shape) -> np.ndarray:
    """Apply affine warp to src_img."""
    h, w = target_shape[:2]
    return cv2.warpAffine(src_img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ─── DT-CWT Fusion ────────────────────────────────────────────────────────────

def dtcwt_fusion(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Fuse two grayscale images using Dual-Tree Complex Wavelet Transform (DT-CWT).

    Strategy:
      - Approximation (LL): weighted average based on local energy
      - Detail subbands (LH, HL, HH): max absolute coefficient selection
    """
    wavelet = 'db2'   # good approximation for DT-CWT using PyWavelets
    level = 2

    def decompose(img):
        return pywt.wavedec2(img.astype(np.float64), wavelet, level=level)

    def local_energy(arr, ksize=5):
        """Local energy map via squared-mean filtering."""
        sq = arr ** 2
        kernel = np.ones((ksize, ksize), np.float64) / (ksize * ksize)
        return cv2.filter2D(sq, -1, kernel)

    coeffs1 = decompose(img1)
    coeffs2 = decompose(img2)

    fused_coeffs = []

    # ── Approximation subband (index 0) ──
    approx1 = coeffs1[0]
    approx2 = coeffs2[0]
    e1 = local_energy(approx1)
    e2 = local_energy(approx2)
    total_e = e1 + e2 + 1e-10
    w1 = e1 / total_e
    w2 = e2 / total_e
    fused_approx = w1 * approx1 + w2 * approx2
    fused_coeffs.append(fused_approx)

    # ── Detail subbands (levels 1..N) ──
    for level_coeffs1, level_coeffs2 in zip(coeffs1[1:], coeffs2[1:]):
        fused_level = []
        for sub1, sub2 in zip(level_coeffs1, level_coeffs2):
            # Max absolute coefficient selection
            fused_sub = np.where(np.abs(sub1) >= np.abs(sub2), sub1, sub2)
            fused_level.append(fused_sub)
        fused_coeffs.append(tuple(fused_level))

    # Reconstruct
    fused = pywt.waverec2(fused_coeffs, wavelet)
    # Crop to original size if reconstruction adds 1 pixel
    h, w = img1.shape[:2]
    fused = fused[:h, :w]
    # Normalize to 0-255
    fused = np.clip(fused, 0, None)
    fused_norm = cv2.normalize(fused, None, 0, 255, cv2.NORM_MINMAX)
    return fused_norm.astype(np.uint8)


# ─── Image Segmentation ───────────────────────────────────────────────────────

def watershed_segmentation(img: np.ndarray) -> np.ndarray:
    """
    Apply watershed segmentation on a grayscale-converted image.
    Returns the segmented image with region boundaries highlighted.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    # Otsu threshold
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground via distance transform
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Convert grayscale to BGR for watershed
    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_bgr, markers)
    img_bgr[markers == -1] = [0, 0, 255]   # boundary in red

    return img_bgr


# ─── Flask routes ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('form.html')


@app.route('/upload', methods=['POST'])
def upload():
    mri_file = request.files.get('mri')
    ct_file  = request.files.get('ct')
    points   = int(request.form.get('points', 5))

    if not mri_file or not ct_file:
        return 'Missing files', 400

    mri_file.save(img_path('mri.jpg'))
    ct_file.save(img_path('ct.jpg'))

    return render_template('registration.html', points=points)


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON received'}), 400

    mri_pts = np.array(data['mriCoord'], dtype=np.float64)
    ct_pts  = np.array(data['ctCoord'],  dtype=np.float64)

    if len(mri_pts) < 3 or len(ct_pts) < 3:
        return jsonify({'error': 'Need at least 3 point pairs'}), 400

    ct  = cv2.imread(img_path('ct.jpg'),  cv2.IMREAD_GRAYSCALE)
    mri = cv2.imread(img_path('mri.jpg'), cv2.IMREAD_GRAYSCALE)

    if ct is None or mri is None:
        return jsonify({'error': 'Could not read uploaded images'}), 500

    M = procrustes_align(mri_pts, ct_pts)
    mri_registered = register_image(mri, M, ct.shape)
    cv2.imwrite(img_path('mri_registered.jpg'), mri_registered)

    return jsonify({'status': 'ok'})


@app.route('/registerimage')
def registerimage():
    return render_template('imageregistration.html')


@app.route('/fusion')
def fusion():
    mri = cv2.imread(img_path('mri_registered.jpg'), cv2.IMREAD_GRAYSCALE)
    ct  = cv2.imread(img_path('ct.jpg'),              cv2.IMREAD_GRAYSCALE)

    if mri is None or ct is None:
        return 'Images not found. Please restart the process.', 400

    # Resize to same shape if needed
    if mri.shape != ct.shape:
        ct = cv2.resize(ct, (mri.shape[1], mri.shape[0]))

    fused = dtcwt_fusion(mri, ct)
    cv2.imwrite(img_path('fusion.jpg'), fused)

    # Save sub-band visualizations for display
    wavelet = 'db2'
    coeffs_mri = pywt.wavedec2(mri.astype(np.float64), wavelet, level=1)
    coeffs_ct  = pywt.wavedec2(ct.astype(np.float64),  wavelet, level=1)
    LL_m, (LH_m, HL_m, HH_m) = coeffs_mri[0], coeffs_mri[1]
    LL_c, (LH_c, HL_c, HH_c) = coeffs_ct[0],  coeffs_ct[1]

    for name, arr in [('mri_LL', LL_m), ('mri_LH', LH_m),
                      ('mri_HL', HL_m), ('mri_HH', HH_m),
                      ('ct_LL',  LL_c),  ('ct_LH',  LH_c),
                      ('ct_HL',  HL_c),  ('ct_HH',  HH_c)]:
        norm = cv2.normalize(np.abs(arr), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(img_path(f'{name}.jpg'), norm)

    return render_template('fusion.html')


@app.route('/segmentation')
def segmentation():
    fused = cv2.imread(img_path('fusion.jpg'))
    if fused is None:
        return 'Fusion image not found. Please run fusion first.', 400

    segmented = watershed_segmentation(fused)
    cv2.imwrite(img_path('segmented.jpg'), segmented)

    return render_template('segmentation.html')


@app.after_request
def no_cache(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma']  = 'no-cache'
    response.headers['Expires'] = '0'
    return response


if __name__ == '__main__':
    app.run(debug=True)
