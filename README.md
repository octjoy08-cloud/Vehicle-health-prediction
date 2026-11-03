# Vehicle Health Intelligence Dashboard

A Streamlit front end for the vehicle-health prediction notebook.

## 1. Install

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Run

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal (normally http://localhost:8501).

## 3. Upload data

Use the left sidebar to upload:
- Telematics: CSV, JSON, JSONL/NDJSON, or Parquet
- CAN-bus (optional): CSV, JSON, JSONL/NDJSON, or Parquet

The JSON loader understands notebook-style outer records with `messages` (telematics) or `messageData` (CAN) arrays. You can also upload already-flattened CSV files.

### Telematics columns
The key field is `vehicleBatteryVoltage`. Useful additional numeric signals include `speed`, `accelX`, `accelY`, `accelZ`, `odometer`, `signalStrength`, `gpsSignalQuality`, and `backupBatteryVoltage`. `vehicleId` and `eventDateTime` are recommended for filtering and time-series plots.

### CAN columns
Long-format CAN data should contain `vin`, `timestamp`, `spnName`, and `spnValue`. Wide-format data can contain `EngSpeed`, `RoadSpeed`, and other SPN columns.

## 4. Dashboard sections

- Overview: fleet KPIs, voltage trend, health distribution, recent records, model-estimated VHI.
- Telematics analytics: signal trend, distribution, and numeric correlation.
- CAN diagnostics: RPM anomaly analysis using the notebook thresholds.
- Predict a vehicle: edit sensor readings and run trained Random Forest models.
- Data & model details: data quality, model metrics, feature list, and CSV export.

## Important modelling note

The notebook defines VHI from battery voltage and defines a fault when battery voltage is below 24.0 V. To avoid direct target leakage, the dashboard excludes `vehicleBatteryVoltage` from the Random Forest predictor features. Consequently, model metrics depend on the uploaded dataset and may differ from the notebook's reported metrics. The CAN RPM thresholds (below 500 and above 2500 RPM) are treated as illustrative rules and should be validated against the vehicle's engineering specification.

This is a research/academic prototype, not a certified diagnostic or safety system.
