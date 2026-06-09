import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

file_path = "/Users/karolinahuang/Desktop/PSY221F_Project/Data/Curly_cell/curly_cell_terminal_v_mv_cleared_df(9).csv"

df = pd.read_csv(file_path)

max_row = df.loc[df["terminal_v_mv"].idxmax()]

print(max_row)

# select data with y = 0
df_y0 = df[df["y"] == -10].copy()

x = df_y0["x"].values
V = df_y0["terminal_v_mv"].values

mask = ~np.isnan(x) & ~np.isnan(V)
x = x[mask]
V = V[mask]

# Gaussian slice at y = 0
# V(x,0) = C + A * exp(-x^2 / (2*sigma^2))
def gaussian_y0(x, A, mu, sigma, C):
    y = 0
    return C + A * np.exp(-((x-mu)**2 + (y-mu)**2) / (2 * sigma**2))

A_guess = V.max() - V.min()
mu_guess = x[np.argmax(V)]
sigma_guess = np.std(x)
C_guess = V.min()

params, covariance = curve_fit(
    gaussian_y0,
    x,
    V,
    p0=[A_guess, mu_guess, sigma_guess, C_guess],
    maxfev=10000
)

A, mu, sigma, C = params
sigma = abs(sigma)

V_pred = gaussian_y0(x, A, mu, sigma, C)

ss_res = np.sum((V - V_pred) ** 2)
ss_tot = np.sum((V - np.mean(V)) ** 2)
r_squared = 1 - ss_res / ss_tot

print("Gaussian fit for y = 0 (df = 9)")
print("-----------------------")
print(f"A = {A:.6f} mV")
print(f"mu = {mu:.6f} um")
print(f"sigma = {sigma:.6f} um")
print(f"C = {C:.6f} mV")
print(f"R squared = {r_squared:.6f}")

print("\nGaussian Model:")
print(f"V(x,0) = {C:.6f} + {A:.6f} * exp(-((x-{mu:.6f})^2 + 0-{mu:.6f})^2)  / (2 * {sigma:.6f}^2))")

# Plotting Gaussian line curve and scatter plot of original data
x_fit = np.linspace(
    x.min(),
    x.max(),
    1000
)

V_fit = gaussian_y0(
    x_fit,
    A,
    mu,
    sigma,
    C
)

plt.figure(figsize=(8,6))

plt.scatter(
    x,
    V,
    s=30,
    alpha=0.7,
    label="Data (y=0)"
)

plt.plot(
    x_fit,
    V_fit,
    color="red",
    linewidth=3,
    label="Gaussian Fit"
)

plt.xlabel("x position (μm)")
plt.ylabel("terminal_v_mv (mV)")
plt.title("Gaussian Fit of Terminal Vm Along y = 0")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()