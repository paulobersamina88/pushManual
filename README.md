# RSA–Pushover Reconciliation Tool

Online Streamlit version for simplified manual MDOF pushover and RSA/base shear comparison.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload these files:
   - app.py
   - requirements.txt
3. Go to Streamlit Community Cloud.
4. Click "New app".
5. Select your GitHub repository.
6. Main file path: app.py
7. Click Deploy.

## Local test

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important modeling note

Use consistent basis:

Mass basis = stiffness basis = plastic moment/yield capacity basis.

If your STAAD floor mass and storey stiffness represent 3 frames in one axis, then the plastic moment capacity must also represent those 3 frames, or use the frame multiplier.
