import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# input file
file_path = "/Users/karolinahuang/Desktop/PSY221F_Project/Data/Curly_cell/curly_cell_terminal_v_mv.csv"

# check if the pathway is right
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found:\n{file_path}")

df = pd.read_csv(file_path)
print(df.head())
print(df.info())

# Checking missing, duplicate data
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:")
print(df.duplicated().sum())

# input variables
numeric_cols = ["time_index", "x", "y", "z", "terminal_v_mv"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna().copy()

# Optional voltage range filter
df = df[
    (df["terminal_v_mv"] > -100) &
    (df["terminal_v_mv"] < 50)
].copy()

V_rest = -45
df["delta_vm"] = np.abs(df["terminal_v_mv"] - V_rest)

print("\nAfter cleaning:")
print(df.info())


# ============================================================
# 4. SELECT TIME POINT AND CLOSEST Z PLANE
# ============================================================

time_point = 5
target_z = 6.15

# Pick closest real z value in the dataset
z_plane = df["z"].iloc[(df["z"] - target_z).abs().argmin()]

print("\nUsing time_index:", time_point)
print("Target z:", target_z)
print("Closest z found:", z_plane)

df_slice = df[
    (df["time_index"] == time_point) &
    (df["z"] == z_plane)
].copy()

print("Rows in selected slice:", len(df_slice))

if df_slice.empty:
    raise ValueError("df_slice is empty. Check your time_index or z values.")


# ============================================================
# 5. DEFINE CIRCULAR REGION
# ============================================================

center_x = -3
center_y = -5
radius = 45

df_slice["distance_from_center"] = np.sqrt(
    (df_slice["x"] - center_x) ** 2 +
    (df_slice["y"] - center_y) ** 2
)

df_circle = df_slice[
    df_slice["distance_from_center"] <= radius
].copy()

df_removed = df_slice[
    df_slice["distance_from_center"] > radius
].copy()

print("\nRows before circular filtering:", len(df_slice))
print("Rows after circular filtering:", len(df_circle))
print("Rows removed:", len(df_removed))

if df_circle.empty:
    raise ValueError(
        "df_circle is empty. Increase radius or check whether x/y values are centered around 0."
    )


# ============================================================
# 6. SAVE CLEANED DATA
# ============================================================

output_path = "/Users/karolinahuang/Desktop/PSY221F_Project/Data/Curly_cell/curly_cell_terminal_v_mv_cleared_df(5).csv"
df_circle.to_csv(output_path, index=False)
print("\nCleaned circular data saved to:")
print(output_path)


# ============================================================
# 7. ORIGINAL VOLTAGE MAP
# ============================================================

full_map = df_slice.pivot_table(
    index="y",
    columns="x",
    values="terminal_v_mv",
    aggfunc="mean"
)

full_map = full_map.sort_index(ascending=True)
full_map = full_map.sort_index(axis=1)

plt.figure(figsize=(6, 5))

im = plt.imshow(
    full_map,
    extent=[
        full_map.columns.min(),
        full_map.columns.max(),
        full_map.index.min(),
        full_map.index.max()
    ],
    origin="lower",
    aspect="equal"
)

plt.colorbar(im, label="Terminal voltage (mV)")
plt.xlabel("x position (μm)")
plt.ylabel("y position (μm)")
plt.title(f"Original voltage map\nz = {z_plane:.2f} μm, time index = {time_point}")

circle = plt.Circle(
    (center_x, center_y),
    radius,
    fill=False,
    color="red",
    linewidth=2
)

plt.gca().add_patch(circle)
plt.show()


# ============================================================
# 8. CHECK WHICH POINTS WERE KEPT OR REMOVED
# ============================================================

plt.figure(figsize=(6, 5))

plt.scatter(
    df_removed["x"],
    df_removed["y"],
    s=10,
    label="Removed outside circle"
)

plt.scatter(
    df_circle["x"],
    df_circle["y"],
    s=10,
    label="Kept inside circle"
)

circle = plt.Circle(
    (center_x, center_y),
    radius,
    fill=False,
    color="black",
    linewidth=2
)

plt.gca().add_patch(circle)
plt.axis("equal")
plt.xlabel("x position (μm)")
plt.ylabel("y position (μm)")
plt.title("Check circular filtering")
plt.legend()
plt.show()


# ============================================================
# 9. VOLTAGE MAP AFTER CIRCULAR FILTERING
# ============================================================

circle_map = df_circle.pivot_table(
    index="y",
    columns="x",
    values="terminal_v_mv",
    aggfunc="mean"
)

circle_map = circle_map.sort_index(ascending=True)
circle_map = circle_map.sort_index(axis=1)

if circle_map.empty:
    raise ValueError("circle_map is empty. Filtering removed all points.")

plt.figure(figsize=(6, 5))

im = plt.imshow(
    circle_map,
    extent=[
        circle_map.columns.min(),
        circle_map.columns.max(),
        circle_map.index.min(),
        circle_map.index.max()
    ],
    origin="lower",
    aspect="equal"
)

plt.colorbar(im, label="Terminal voltage (mV)")
plt.xlabel("x position (μm)")
plt.ylabel("y position (μm)")
plt.title(
    f"Voltage map after circular filtering\n"
    f"radius = {radius} μm, z = {z_plane:.2f} μm"
)

circle = plt.Circle(
    (center_x, center_y),
    radius,
    fill=False,
    color="red",
    linewidth=2
)

plt.gca().add_patch(circle)
plt.show()