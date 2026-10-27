"""
Crime Pattern Analysis and Forecasting: reproducible demonstration.
IMPORTANT: This project uses SYNTHETIC data generated for demonstration.
It does not establish real-world crime rates or operational predictive performance.
Run: pip install pandas numpy matplotlib scikit-learn
     python crime_analysis.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.linear_model import LinearRegression

SEED = 42
OUT = "outputs"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(SEED)

# Synthetic observations: these are NOT real police records.
n = 360
regions = rng.choice(["Region_A", "Region_B", "Region_C", "Region_D", "Region_E", "Region_F"], n)
month = rng.integers(1, 13, n)
population = rng.integers(8000, 90000, n)
urban_index = rng.uniform(0.1, 1.0, n)
prior_incidents = rng.poisson(4 + urban_index * 8, n)
arrests = rng.poisson(1 + prior_incidents * 0.28, n)
theft = rng.poisson(1.2 + urban_index * 2.5 + prior_incidents * 0.22, n)
burglary = rng.poisson(0.6 + urban_index * 1.4 + prior_incidents * 0.18, n)
robbery = rng.poisson(0.3 + urban_index * 0.9 + prior_incidents * 0.10, n)
vehicle_theft = rng.poisson(0.5 + urban_index * 1.2 + prior_incidents * 0.13, n)
total_crime = theft + burglary + robbery + vehicle_theft
hotspot = (total_crime >= np.quantile(total_crime, 0.65)).astype(int)

df = pd.DataFrame({
    "region": regions, "month": month, "population": population,
    "urban_index": urban_index.round(3), "prior_incidents": prior_incidents,
    "arrests": arrests, "theft": theft, "burglary": burglary,
    "robbery": robbery, "vehicle_theft": vehicle_theft,
    "total_crime": total_crime, "hotspot": hotspot
})
df.to_csv(os.path.join(OUT, "synthetic_crime_data.csv"), index=False)

# 1. K-Means: standardize count features before clustering.
cluster_features = ["prior_incidents", "theft", "burglary", "robbery", "vehicle_theft"]
scaled = StandardScaler().fit_transform(df[cluster_features])
kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=10)
df["cluster"] = kmeans.fit_predict(scaled)
df.groupby("cluster")[cluster_features + ["total_crime"]].mean().to_csv(
    os.path.join(OUT, "cluster_summary.csv")
)

# 2. Decision Tree classification with a held-out test split.
features = ["month", "population", "urban_index", "prior_incidents", "arrests",
            "theft", "burglary", "robbery", "vehicle_theft"]
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["hotspot"], test_size=0.25, random_state=SEED,
    stratify=df["hotspot"]
)
model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5,
                               random_state=SEED, class_weight="balanced")
model.fit(X_train, y_train)
pred = model.predict(X_test)
metrics = {
    "Accuracy": accuracy_score(y_test, pred),
    "Precision": precision_score(y_test, pred, zero_division=0),
    "Recall": recall_score(y_test, pred, zero_division=0),
    "F1-score": f1_score(y_test, pred, zero_division=0)
}
pd.DataFrame([metrics]).to_csv(os.path.join(OUT, "classification_metrics.csv"), index=False)

# 3. Linear regression demonstration for count estimation.
reg = LinearRegression()
reg.fit(df[["prior_incidents", "population", "urban_index"]], df["total_crime"])
reg_pred = reg.predict(df[["prior_incidents", "population", "urban_index"]])
reg_metrics = {
    "MAE": mean_absolute_error(df["total_crime"], reg_pred),
    "RMSE": mean_squared_error(df["total_crime"], reg_pred) ** 0.5,
    "R2": r2_score(df["total_crime"], reg_pred)
}
pd.DataFrame([reg_metrics]).to_csv(os.path.join(OUT, "regression_metrics.csv"), index=False)
df.to_csv(os.path.join(OUT, "crime_analysis_output.csv"), index=False)

# Figures
plt.figure(figsize=(7, 4.5))
df["total_crime"].hist(bins=15)
plt.title("Distribution of Synthetic Incident Counts")
plt.xlabel("Total incident count (synthetic)")
plt.ylabel("Number of records")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "crime_distribution.png"), dpi=180)
plt.close()

plt.figure(figsize=(7, 4.5))
for c in sorted(df["cluster"].unique()):
    s = df[df["cluster"] == c]
    plt.scatter(s["prior_incidents"], s["total_crime"], label=f"Cluster {c}", alpha=0.7)
plt.title("K-Means Clusters (Synthetic Data)")
plt.xlabel("Prior incidents")
plt.ylabel("Total incident count")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "kmeans_clusters.png"), dpi=180)
plt.close()

ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                       display_labels=["Non-hotspot", "Hotspot"]).plot(values_format="d")
plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "confusion_matrix.png"), dpi=180)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.bar(list(metrics.keys()), list(metrics.values()))
plt.ylim(0, 1.05)
plt.title("Decision Tree Evaluation Metrics")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "classification_metrics.png"), dpi=180)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.scatter(df["total_crime"], reg_pred, alpha=0.7)
lims = [min(df["total_crime"].min(), reg_pred.min()),
        max(df["total_crime"].max(), reg_pred.max())]
plt.plot(lims, lims)
plt.title("Regression: Actual vs Predicted Counts")
plt.xlabel("Actual total count")
plt.ylabel("Predicted total count")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "regression_actual_predicted.png"), dpi=180)
plt.close()

print("Completed. All data, metrics, and figures are in the outputs/ directory.")
print("Classification metrics:", metrics)
print("Regression metrics:", reg_metrics)
print("Reminder: results are from synthetic demonstration data, not real crime records.")
