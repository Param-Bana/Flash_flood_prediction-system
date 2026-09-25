
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

BASE = Path(".")
events = pd.read_csv(BASE / "floodevents_indofloods.csv")
precip = pd.read_csv(BASE / "precipitation_variables_indofloods.csv")
catchment = pd.read_csv(BASE / "catchment_characteristics_indofloods.csv")
metadata = pd.read_csv(BASE / "metadata_indofloods.csv")

events["GaugeID"] = events["EventID"].str.rsplit("-", n=1).str[0]

df = (
    events
    .merge(precip, on="EventID", how="inner")
    .merge(catchment, on="GaugeID", how="inner")
    .merge(metadata, on="GaugeID", how="left", suffixes=("", "_meta"))
)

df["target_severe_flood"] = (
    df["Flood Type"].astype(str).str.strip() == "Severe Flood"
).astype(int)

FEATURES = [f"T{i}d" for i in range(1, 11)] + [
    "Stream Order",
    "Drainage Area",
    "Catchment Relief",
    "Catchment Length",
    "Drainage Density",
    "Ruggedness Number",
    "Annual Precipitation",
]

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = df["target_severe_flood"]
groups = df["GaugeID"]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("rf", RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X.iloc[train_idx], y.iloc[train_idx])

bundle = {
    "pipeline": model,
    "features": FEATURES,
    "target": "target_severe_flood",
    "target_description": (
        "1 = Severe Flood, 0 = Flood in INDOFLOODS. "
        "Prototype proxy, not a direct flash-flood label."
    )
}

joblib.dump(bundle, "sih26192_flash_flood_rf_prototype.joblib", compress=3)
print("Saved sih26192_flash_flood_rf_prototype.joblib")
