# Home Sentry

Home Sentry is an experimental wireless intrusion-detection project that captures 802.11 traffic with TShark, engineers protocol and timing features, and evaluates 20-packet sequences with a PyTorch GRU. A Flask dashboard provides network selection, live status, detection counts, and forensic context.

> **Authorized environments only.** Home Sentry is intended for defensive research and testing on networks you own or are explicitly authorized to monitor.

## Highlights

- Live 802.11 capture through TShark
- 58-feature packet representation including WLAN, radiotap, timing, and engineered behavioral features
- 20-packet GRU sequence classification
- Saved scikit-learn preprocessing pipeline
- Flask dashboard for live monitoring
- Forensic snapshot logging around high-confidence detections

## Repository layout

- `app.py` - Flask application and process control
- `home_sentry/live_detector.py` - live capture, preprocessing, inference, and forensic logging
- `home_sentry/model.py` - shared GRU architecture
- `home_sentry/features.py` - shared feature schema
- `training/` - data collection, preprocessing, labeling, scaler construction, training, and evaluation utilities
- `models/` - trained model weights and preprocessing artifact
- `templates/dashboard.html` - monitoring UI

## Requirements

Python dependencies are listed in `requirements.txt`. Live capture additionally requires Linux wireless monitor mode and TShark installed on the host.

## Running the dashboard

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`. The default monitor interface is `wlan0mon`; override it with the `HOME_SENTRY_INTERFACE` environment variable.

## Model pipeline

The repository preserves the project workflow from capture and preprocessing through scaling, GRU training, offline evaluation, and live inference. Raw packet captures and generated forensic logs are intentionally excluded from version control.

## Notes

This repository is a cleaned portfolio version of an academic/research project. Dataset paths in some training utilities may need to be adapted to your local dataset layout before rerunning the full training pipeline.
