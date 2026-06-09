import pandas as pd
import numpy as np

# Load raw data
df = pd.read_csv("/Users/karolinahuang/Desktop/PSY221F_Project/Data/Curly_cell/curly_cell_terminal_v_mv.csv")

# Circle mask: center (0,0), radius 50
df["distance_from_center"] = np.sqrt(df["x"]**2 + df["y"]**2)
df["inside_circle"] = df["distance_from_center"] <= 50

# Keep only points inside circle
df_inside = df[df["inside_circle"]].copy()

# Extract same x, y, z location across different time_index
wide_df = df_inside.pivot_table(
    index=["x", "y", "z"],
    columns="time_index",
    values="terminal_v_mv",
    aggfunc="mean"
).reset_index()

# Rename time columns
wide_df.columns = [
    col if col in ["x", "y", "z"]
    else f"Vm_T{int(col)+1}"
    for col in wide_df.columns
]

# Save result
output_path = "/Users/karolinahuang/Desktop/PSY221F_Project/Data/Curly_cell/curly_cell_terminal_v_changes_AmongTime.csv"
wide_df.to_csv(output_path, index=False)