# SIH26192 Random Forest Prototype

## What this prototype does

This prototype uses the actual INDOFLOODS database files:

- floodevents_indofloods.csv
- precipitation_variables_indofloods.csv
- catchment_characteristics_indofloods.csv
- metadata_indofloods.csv

The model is a Random Forest classifier.

## Important target limitation

The INDOFLOODS event table contains two labels:

- Flood
- Severe Flood

The prototype predicts **Severe Flood vs Flood**.

This is a **proxy for higher flood severity**, not a direct flash-flood occurrence label.

Therefore the prototype must NOT be presented as a validated flash-flood early-warning model.

## Leakage control

The following post-event variables were deliberately NOT used as predictors:

- Peak Flood Level
- Peak Flood Date
- Peak Discharge
- Peak Discharge Date
- Flood Volume
- Event Duration
- Time to Peak
- Recession Time

Events were split by GaugeID rather than randomly by row so that events from the same gauge do not appear in both train and test sets.

## Features

Rainfall:
T1d, T2d, T3d, T4d, T5d, T6d, T7d, T8d, T9d, T10d

Catchment:
Stream Order, Drainage Area, Catchment Relief, Catchment Length, Drainage Density, Ruggedness Number, Annual Precipitation

## Prototype results

Rows: 4548
Train rows: 3760
Test rows: 788
Gauges: 155

Accuracy: 0.6688
Precision (Severe Flood): 0.5315
Recall (Severe Flood): 0.4874
F1 (Severe Flood): 0.5085
ROC-AUC: 0.6577
PR-AUC: 0.5259

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The Streamlit interface simulates field/sensor inputs. It does not connect to real sensors.

## Next ML step

Replace the Severe Flood proxy with a time-indexed flash-flood occurrence target and add real dynamic observations such as soil moisture, water level/discharge and forecast rainfall.
