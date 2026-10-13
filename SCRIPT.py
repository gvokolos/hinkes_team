import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 1. Read and clean the data
df = pd.read_csv('Data_Tower_projects_1-4_8oct2026_/profile.txt', header=None)
cols_to_check_negatives = [5, 6, 7, 8, 9, 15, 17, 19, 21, 23]
df_cleaned = df.dropna().copy()
for col in cols_to_check_negatives:
    df_cleaned = df_cleaned[df_cleaned[col] >= 0]

# 2. Extract DOY and calculate daily means
df_cleaned['Day'] = df_cleaned[1].astype(int)
daily_means = df_cleaned.groupby('Day').mean()

# Define measurement heights and smooth array for continuous fit lines
heights = np.array([8.0, 4.0, 2.0, 1.0, 0.5])
log_heights = np.log(heights)
smooth_heights = np.linspace(0.5, 8.0, 100)
log_smooth_heights = np.log(smooth_heights)

# Create 3 subplots: Wind Speed, Temperature, and Wind Direction
fig, axs = plt.subplots(1, 3, figsize=(18, 7))

# Initialize lists to store daily parameters for the overall means
wind_slopes, wind_intercepts = [], []
temp_slopes, temp_intercepts = [], []

# 3. Calculate and plot fits for each individual DOY
for day in daily_means.index:
    # Extract daily values
    day_winds = daily_means.loc[day, [15, 17, 19, 21, 23]].values
    day_temps = daily_means.loc[day, 10:14].values
    day_dirs = daily_means.loc[day, [16, 18, 20, 22, 24]].values

    # Wind Speed Fit
    w_slope, w_intercept, w_r, _, _ = linregress(log_heights, day_winds)
    wind_slopes.append(w_slope)
    wind_intercepts.append(w_intercept)
    fitted_winds = w_slope * log_smooth_heights + w_intercept

    # Plot Wind raw points and extract auto-assigned color
    w_points = axs[0].plot(day_winds, heights, marker='o', linestyle='', alpha=0.5)
    line_color = w_points[0].get_color()
    axs[0].plot(fitted_winds, smooth_heights, linestyle='-', color=line_color, alpha=0.7,
                label=f'DOY {day} ($R^2$={w_r ** 2:.3f})')

    # Temperature Fit
    t_slope, t_intercept, t_r, _, _ = linregress(log_heights, day_temps)
    temp_slopes.append(t_slope)
    temp_intercepts.append(t_intercept)
    fitted_temps = t_slope * log_smooth_heights + t_intercept

    # Plot Temperature raw points and fit using the same color
    axs[1].plot(day_temps, heights, marker='o', linestyle='', color=line_color, alpha=0.5)
    axs[1].plot(fitted_temps, smooth_heights, linestyle='-', color=line_color, alpha=0.7,
                label=f'DOY {day} ($R^2$={t_r ** 2:.3f})')

    # Plot Wind Direction (no log-fit needed for angles, just connecting points)
    axs[2].plot(day_dirs, heights, marker='o', linestyle='-', color=line_color, alpha=0.7, label=f'DOY {day}')

# 4. Calculate and plot overall means
# Wind Overall Mean
mean_winds = daily_means.loc[:, [15, 17, 19, 21, 23]].mean().values
mean_w_slope = np.mean(wind_slopes)
mean_w_intercept = np.mean(wind_intercepts)
mean_fitted_winds = mean_w_slope * log_smooth_heights + mean_w_intercept

axs[0].plot(mean_winds, heights, marker='s', markersize=8, color='black', linestyle='', label='Mean Data')
axs[0].plot(mean_fitted_winds, smooth_heights, linestyle='--', color='black', linewidth=3, label='Mean Fit')

# Temperature Overall Mean
mean_temps = daily_means.loc[:, 10:14].mean().values
mean_t_slope = np.mean(temp_slopes)
mean_t_intercept = np.mean(temp_intercepts)
mean_fitted_temps = mean_t_slope * log_smooth_heights + mean_t_intercept

axs[1].plot(mean_temps, heights, marker='s', markersize=8, color='black', linestyle='', label='Mean Data')
axs[1].plot(mean_fitted_temps, smooth_heights, linestyle='--', color='black', linewidth=3, label='Mean Fit')

# Wind Direction Overall Mean
mean_dirs = daily_means.loc[:, [16, 18, 20, 22, 24]].mean().values
axs[2].plot(mean_dirs, heights, marker='s', markersize=8, color='black', linewidth=3, linestyle='--',
            label='Overall Mean')

# 5. Format plots
titles = ['Wind Speed Log-Profile Fits', 'Temperature Log-Profile Fits', 'Mean Wind Direction Profile']
xlabels = ['Wind Speed (m/s)', 'Temperature (°C)', 'Wind Direction (°)']

for i, ax in enumerate(axs):
    ax.set_title(titles[i])
    ax.set_xlabel(xlabels[i])
    ax.set_ylabel('Height (m) - Log Scale')
    ax.set_yscale('log')
    ax.set_yticks(heights)
    ax.set_yticks([], minor=True)
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, ls="--")
    if i < 2:
        ax.legend(fontsize='small', loc='best')
    else:
        # Move legend outside for the last plot to save space
        ax.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()