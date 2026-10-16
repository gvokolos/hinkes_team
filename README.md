# hinkes_team
You can find two files. The python script (.py), a draft of the proposal (.pdf) and (.zip). Below you can find some notes.

To run the script on a different dataset folder, only change `FOLDER_PATH` at the top of the file. All figures and CSV tables are automatically saved inside the `analysis_output/` folder.

## 1. How the Code Works (Step-by-Step)

1. **Loads Raw Data & Audits Errors (`Table 1`):**
   * Reads `profile.txt` (30-min mast data), `profile1min.txt` (1-min mast data), `cnr1.txt` (radiation data), `cnr1raw.txt`, and `iws.txt` (weather station data).
   * Converts the Year and Decimal Day of Year (DOY) into standard dates and timestamps.
   * Counts missing (`NaN`) and corrupted/negative rows in every file and saves the summary to `stats_1_file_dates_and_errors.csv`.

2. **Applies Proposal Filters (`Table 2` & `Table 3`):**
   * **Step 1 (Missing data):** Removes rows where any sensor has a `NaN` value (for example, wind sensors were offline during DOY 258 and the morning of DOY 259).
   * **Step 2 (Invalid values):** Removes negative wind speeds, negative relative humidity (RH), or unrealistic temperatures (`< -40 °C`). If RH is slightly above `100%` (e.g., `100.1%` during night fog due to sensor tolerance), it caps RH at `100.0%` instead of deleting the row.
   * **Step 3 (North wind filter):** Removes rows where wind blows from the North (`315°–360°` or `0°–45°`) at the 8.0 m sensor to avoid wind distortion caused by the mast structure itself.
   * **Step 4 (Low wind speed filter):** Checks both `WS > 2.0 m/s` and `WS > 1.0 m/s` across all 5 heights and saves the comparison in `stats_2_filtering_breakdown.csv` and `stats_3_daily_doy_counts.csv`.

3. **Creates Figures & Profile Fits (`Table 4` & `Table 5`):**
   * Plots time series, logarithmic wind and temperature profiles, stability profiles, fetch direction profiles, and surface heat fluxes (`H` and `LE`), and saves all fit parameters (`R²`, `u*`, `z0`, `dT/dz`) to `stats_4_profile_fits_summary.csv`.

---

## 2. Main Assumptions in the Code

* **Sensor Heights:** The 5 mast levels are ordered from top to bottom in the raw file: `8.0 m, 4.0 m, 2.0 m, 1.0 m, and 0.5 m`.
* **Neutral Logarithmic Profile:** Under near-neutral conditions, wind speed and temperature vary logarithmically with height (`ln(z)`), using the von Karman constant `κ = 0.40`.
* **Zero-Plane Displacement (`d = 0`):** Because the mast is over short grass, the displacement height `d` is assumed to be `0 m`.
* **Stability Classification by Temperature Gradient:** Atmospheric stability is found using only the temperature difference between the top and bottom of the mast (`dT = T_8.0m - T_0.5m`):
  * **Unstable:** `dT < -0.10 °C` (air is warmer near the ground during the day).
  * **Neutral:** `-0.10 °C <= dT <= +0.10 °C` (temperature is nearly constant with height).
  * **Stable:** `dT > +0.10 °C` (air is cooler near the ground at night).
* **Nighttime Radiation Offset:** Small negative incoming shortwave radiation values at night (`-3 to 0 W/m²`) are normal thermal offsets in pyranometers, so only corrupted spikes (`< -10 W/m²`) or `NaN` net radiation rows are removed from `cnr1.txt`.
* **Constant Air Properties for Fluxes:** Air density (`ρ = 1.20 kg/m³`), specific heat (`cp = 1005 J/(kg·K)`), latent heat of vaporization (`Lv = 2.45 × 10⁶ J/kg`), and mean pressure (`101.3 kPa`) are treated as constants.

---

## 3. Team Q&A: Explaining Our Results & Figures

### Q1: Did the plots use `WS > 2.0 m/s` or `WS > 1.0 m/s`, and why?
* The script calculates statistics for **both**, but the plots use **`WS > 1.0 m/s`**.
* **Why:** Wind is much slower at the lowest sensor (`0.5 m`) than at `8.0 m`. If we require `WS > 2.0 m/s` at `0.5 m`, we lose 116 extra rows and are left with **only 1 data point on DOY 259** and **only 1 data point on DOY 264**, plus we lose almost all nighttime data. Using `WS > 1.0 m/s` (which is the physical starting threshold of the Young 05103 wind sensor) keeps enough data on every day (`DOY 259–264`) to make daily plots.

### Q2: In Figure 2, why is the black "Mean Data" line (~2.5 m/s at 0.5 m) higher than the middle of the daily lines?
* The black line is the average of **all 168 kept 30-minute timestamps**, not the average of the 6 daily lines.
* Calm days (`DOY 259` and `DOY 264`) only have `10–13` valid timestamps each, while windy days (`DOY 261, 262, 263`) have `35–48` timestamps each. Because most of the valid timestamps come from windy days, the overall average is pulled toward the higher wind speeds.

### Q3: How is Figure 3 (Wind & Temperature Profiles by Stability) made, and what days are included?
* **Days included:** All valid days combined (`DOY 259–264`).
* **How it works:** Every 30-minute row is grouped into **Unstable (Red)**, **Neutral (Green)**, or **Stable (Blue)** using only the temperature difference `T_8.0m - T_0.5m`.
* **Right graph (Temperature):** Plots `T(z) - T(0.5m)` against linear height `z`. Subtracting `T(0.5m)` makes all three lines start at `0 °C` at the bottom so we can easily compare their slopes:
  * **Unstable (Red)** tilts left (gets colder with height during the day).
  * **Neutral (Green)** goes straight up near `0 °C`.
  * **Stable (Blue)** tilts right (gets warmer with height at night).
* **Left graph (Wind Speed):** Shows the average wind profile for those exact same three temperature-stability groups on a log-height axis (`ln(z)`).

### Q4: Does Figure 1 (Daily Evolution) still have North winds in it?
* **No.** Figure 1 is plotted after all filters are applied, so `NaN` values, North winds (`315°–45°`), and low wind speeds (`<= 1.0 m/s`) are already removed. That is why there are gaps in the curves.

### What is Figure 4 (Fetch Sector Profiles), and do we need it?
* "Fetch" is the upwind surface area before the wind hits the mast. Figure 4 splits the wind profiles into 4 wind direction sectors (`East`, `South`, `SW–West`, `WNW`) to check if obstacles in different directions change the roughness length (`z0`) and fit quality (`R²`).
