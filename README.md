# Medical Image Fusion — DT-CWT

A Flask web application for fusing CT and MRI brain images using the
**Dual-Tree Complex Wavelet Transform (DT-CWT)** method, as described in:

> *Medical Image Fusion using Dual-Tree Complex Wavelet Transform for CT and MRI Modalities*  
> Published in IJEAST

---

## Pipeline

```
Upload MRI + CT
      ↓
Landmark-based Affine Registration   (user clicks N point pairs)
      ↓
DT-CWT Fusion                        (wavelet decomp → coefficient fusion → reconstruct)
      ↓
Watershed Segmentation               (Otsu + morphology + watershed)
```

---

## Setup

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## Project Structure

```
image_fusion/
├── app.py                  # Main Flask app + all image processing logic
├── requirements.txt
├── static/
│   ├── css/main.css        # Styles
│   └── images/             # Auto-created; stores uploaded & processed images
└── templates/
    ├── base.html
    ├── form.html            # Step 1: Upload
    ├── registration.html    # Step 2: Landmark selection
    ├── imageregistration.html  # Step 2 result
    ├── fusion.html          # Step 3: Fusion result + sub-bands
    └── segmentation.html    # Step 4: Segmentation result
```

---

## Key Technical Notes

### DT-CWT Fusion (`dtcwt_fusion` in app.py)
- Uses `pywt.wavedec2` with `db2` wavelet (best PyWavelets approximation of DT-CWT)
- **Approximation subband (LL):** energy-weighted average using local energy maps
- **Detail subbands (LH, HL, HH):** max absolute coefficient selection
- 2-level decomposition for multi-scale feature capture

### Registration (`procrustes_align` in app.py)
- Least-squares affine transform from N user-clicked landmark pairs
- `cv2.warpAffine` applies the transform to align MRI → CT space

### Segmentation (`watershed_segmentation` in app.py)
- Otsu thresholding → morphological opening → distance transform
- Watershed markers → boundary detection (red overlay)
