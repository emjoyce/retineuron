import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# input data
# Simple cell
# file_T5 = "/Users/karolinahuang/Desktop/PSY221F_Project/Data/Normal_cell/terminal_v_mv_cleared_df(5).csv"
# file_T6 = "/Users/karolinahuang/Desktop/PSY221F_Project/Data/Normal_cell/terminal_v_mv_cleared_df(6).csv"
# file_T7 = "/Users/karolinahuang/Desktop/PSY221F_Project/Data/Normal_cell/terminal_v_mv_cleared_df(7).csv"
# file_T8 = "/Users/karolinahuang/Desktop/PSY221F_Project/Data/Normal_cell/terminal_v_mv_cleared_df(8).csv"
# file_T9 = "/Users/karolinahuang/Desktop/PSY221F_Project/Data/Normal_cell/terminal_v_mv_cleared_df(9).csv"

# Complex Cell
# file_T5 = "/Users/karolinahuang/Desktop/PSY221F_Project/Data/Curly_cell/curly_cell_terminal_v_mv_cleared_df(5).csv"
# file_T7 = "/Users/karolinahuang/Desktop/PSY221F_Project/Data/Curly_cell/curly_cell_terminal_v_mv_cleared_df(7).csv"
file_T9_S = "/Users/karolinahuang/Desktop/PSY221F_Project/Data/Normal_cell/terminal_v_mv_cleared_df(9).csv"
file_T9_C = "/Users/karolinahuang/Desktop/PSY221F_Project/Data/Curly_cell/curly_cell_terminal_v_mv_cleared_df(9).csv"

df_T9_S = pd.read_csv(file_T9_S)
df_T9_C = pd.read_csv(file_T9_C)

# df_T5 = pd.read_csv(file_T5)
# df_T6 = pd.read_csv(file_T6)
# df_T7 = pd.read_csv(file_T7)
# df_T8 = pd.read_csv(file_T8)
# df_T9 = pd.read_csv(file_T9)


# Gaussian fit parameters for y = 0, df = 9
# Simple Cell
# A = 87.522762
# mu = -0.939399
# sigma = 19.873718
# C = -66.090260

# Complex Cell
A = 34.317125
mu = -4.369663
sigma = 21.963793
C = -55.807797

# Gaussian model
def gaussian(x):
    return C + A * np.exp(-((x - mu)**2) / (2 * sigma**2))

x = np.linspace(-50, 50, 500)
y = gaussian(x)

Plot
plt.figure(figsize=(8, 6))

plt.plot(
    x,
    y,
    color="red",
    linewidth=3,
    label=f"Fitted Gaussian Curve \n σ = 21.963793 ; μ = -4.369663"
)

plt.xlabel("x position (µm)")
plt.ylabel("Terminal Voltage (mV)")
plt.title("Gaussian Fit of Terminal Voltage Along x-axis (y = 0)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# Multiple scatter plot
plt.figure(figsize=(8,6))

plt.scatter(
    df_T5["x"],
    df_T5["terminal_v_mv"],
    s=15,
    alpha=0.5,
    label="Time Point 5"
)

# plt.scatter(
#     df_T6["x"],
#     df_T6["terminal_v_mv"],
#     s=15,
#     alpha=0.5,
#     label="Time Point 6"
# )

plt.scatter(
    df_T7["x"],
    df_T7["terminal_v_mv"],
    s=15,
    alpha=0.5,
    label="Time Point 7"
)

# plt.scatter(
#     df_T8["x"],
#     df_T8["terminal_v_mv"],
#     s=15,
#     alpha=0.5,
#     label="Time Point 8"
# )

plt.scatter(
    df_T9["x"],
    df_T9["terminal_v_mv"],
    s=15,
    alpha=0.5,
    label="Time Point 9"
)

plt.xlabel("x position (μm)")
plt.ylabel("Terminal Voltage (mV)")
plt.title("Distribution of Terminal Voltage of Complex Morphology \nCell in Different x Locations over Time")

plt.legend(fontsize=8)
plt.grid(True)

plt.tight_layout()
plt.show()

# 3D surface plot

z_column = "terminal_v_mv"

def make_surface(data, value_col):
    surface = data.pivot_table(
        index="y",
        columns="x",
        values=value_col,
        aggfunc="mean"
    )

    surface = surface.sort_index()
    surface = surface.sort_index(axis=1)

    X, Y = np.meshgrid(
        surface.columns.values,
        surface.index.values
    )

    Z = surface.values

    return X, Y, Z


X1, Y1, Z1 = make_surface(df_T9_S, z_column)
X2, Y2, Z2 = make_surface(df_T9_C, z_column)

print("Cell 1 max =", np.nanmax(Z1))
print("Cell 1 min =", np.nanmin(Z1))

print("Cell 2 max =", np.nanmax(Z2))
print("Cell 2 min =", np.nanmin(Z2))


from matplotlib.patches import Patch

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Surface 1
ax.plot_surface(
    X1, Y1, Z1,
    cmap="viridis",
    alpha=0.7
)

# Surface 2
ax.plot_surface(
    X2, Y2, Z2,
    cmap="plasma",
    alpha=0.7
)

# Create legend entries
legend_elements = [
    Patch(facecolor='green', label='Simple Morphology Cell'),
    Patch(facecolor='orange', label='Complex Morphology Cell')
]

ax.legend(handles=legend_elements, loc='upper right')

ax.set_xlabel("x position (µm)")
ax.set_ylabel("y position (µm)")
ax.set_zlabel("Terminal Voltage (mV)")

plt.show()

# Looking into how terimal voltage change respect to different time points
file_T = "/Users/karolinahuang/Desktop/PSY221F_Project/Data/Curly_cell/curly_cell_terminal_v_changes_AmongTime.csv"
df_T = pd.read_csv(file_T)

# Select first cell and plot the line graph
row = df_T.iloc[816]
time_points = range(1, 16)
vm_values = [
    row[f"Vm_T{i}"]
    for i in time_points
]

plt.plot(time_points, vm_values, marker="o")
plt.xlabel("Time Index")
plt.ylabel("Terminal Vm (mV)")
plt.title(
    f"Cell at ({row['x']:.1f}, {row['y']:.1f}, {row['z']:.1f})"
)
plt.grid(True)
plt.show()

# plot out the line graph of multiple cell's terminal voltage for different time point and colored according to the distance
df_T["distance_from_center"] = np.sqrt(df_T["x"]**2 + df_T["y"]**2)

vm_cols = sorted(
    [c for c in df_T.columns if c.startswith("Vm_T")],
    key=lambda x: int(x.split("T")[1])
)

time_points = np.arange(1, len(vm_cols) + 1)

norm = Normalize(
    vmin=df_T["distance_from_center"].min(),
    vmax=df_T["distance_from_center"].max()
)

cmap = plt.cm.viridis

fig, ax = plt.subplots(figsize=(10, 6))

for _, row in df_T.iterrows():
    color = cmap(norm(row["distance_from_center"]))

    ax.plot(
        time_points,
        row[vm_cols].values,
        color=color,
        linewidth=0.8,
        alpha=0.8
    )

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Distance of the Dendritic Tip of Cell from Center (μm)")

ax.set_xlabel("Time Index")
ax.set_ylabel("Terminal Vm (mV)")
ax.set_title("Temporal Change of Terminal Voltage for Simple Morphology Cell")
ax.grid(True)

plt.show()