import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("/Users/karolinahuang/Desktop/PSY221F_Project/Data/Normal_cell/terminal_v_mv_cleared_df(9).csv")

# Scatter plot
plt.figure(figsize=(8,6))

plt.scatter(
    df["distance_from_center"],
    df["terminal_v_mv"],
    alpha=0.4,
    s=20,
    label="Cell"
)

x_fit = np.linspace(
    df["distance_from_center"].min(),
    df["distance_from_center"].max(),
    500
)
y_fit = 22.076 - 1.868 * x_fit

plt.plot(
    x_fit,
    y_fit,
    color = "red",
    linewidth=3,
    label= "Least-squares regression line (R² = 0.966)"
)

plt.xlabel("Distance of the Dendritic Tip of Cell from Electrode Center (µm)", fontsize=12)
plt.ylabel("Terminal Voltage (mV)", fontsize=12)
plt.title(
    "Relationship Between Distance of the Dendritic Tip of Complex \nMorphology Cells from Electrode Center and Terminal Voltage",
    fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()