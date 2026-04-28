# Manual RSA–Pushover Reconciliation Tool

Offline Streamlit teaching app for simplified MDOF shear-building pushover analysis.

## How to run offline

1. Install Python 3.10 or newer.
2. Open terminal inside this folder.
3. Install requirements:

```bash
pip install -r requirements.txt
```

4. Run:

```bash
streamlit run app.py
```

## What it does

- Uses floor weights and storey stiffness extracted from STAAD.
- Computes modal properties and first-mode pushover force pattern.
- Estimates yield capacity per storey from beam and column plastic moments.
- Generates a simplified bilinear MDOF pushover curve.
- Compares static or dynamic base shear demand against first yield.

## Important modeling reminder

Use a consistent basis:

Mass basis = stiffness basis = plastic moment/yield capacity basis.

If STAAD mass and stiffness represent 3 frames in one axis, plastic moment capacity must also represent the same 3 frames, or use the app's frame multiplier.
