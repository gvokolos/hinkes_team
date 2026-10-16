import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

FOLDER_PATH = "Data_Tower_projects_1-4_8oct2026_"
OUTPUT_DIR = os.path.join(FOLDER_PATH, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEIGHTS = np.array([8.0, 4.0, 2.0, 1.0, 0.5])

# FILTERS
NORTH_SECTOR_DEG = (315.0, 45.0)
WS_PRIMARY_LIMIT = 2.0
WS_FALLBACK_LIMIT = 1.0
MIN_ROWS_PER_DAY = 5
DT_NEUTRAL_BAND = 0.10

# CONSTANTS
KAPPA = 0.40
GAMMA_D = 0.0098
RHO_AIR = 1.20
CP_AIR = 1005.0
LV_WATER = 2.45e6


def doy_to_datetime(year, decimal_doy):
    dt = datetime(int(year), 1, 1) + timedelta(days=float(decimal_doy) - 1.0)
    return pd.Timestamp(dt).round("10s")


def load_profile_data(filepath):
    cols = [
        "Year", "Decimal_DOY", "HHMM", "Batt_V", "T_Logger",
        "RH_8.0m", "RH_4.0m", "RH_2.0m", "RH_1.0m", "RH_0.5m",
        "T_8.0m", "T_4.0m", "T_2.0m", "T_1.0m", "T_0.5m",
        "WS_8.0m", "WD_8.0m",
        "WS_4.0m", "WD_4.0m",
        "WS_2.0m", "WD_2.0m",
        "WS_1.0m", "WD_1.0m",
        "WS_0.5m", "WD_0.5m"
    ]
    df = pd.read_csv(filepath, header=None, names=cols, na_values=["NaN", "nan", "Inf", "-Inf"])
    df["Datetime"] = [doy_to_datetime(y, d) for y, d in zip(df["Year"], df["Decimal_DOY"])]
    df["DOY"] = df["Decimal_DOY"].astype(int)
    return df


def load_cnr1_data(filepath):
    cols = [
        "Year", "Decimal_DOY", "HHMM",
        "SW_in", "SW_out", "LW_in_raw", "LW_out_raw",
        "T_body_C", "T_body_K", "LW_in", "LW_out",
        "SW_net", "LW_net", "Albedo", "R_net", "LW_in_biased", "LW_out_biased"
    ]
    df = pd.read_csv(filepath, header=None, names=cols, na_values=["NaN", "nan", "Inf", "-Inf"])
    df["Datetime"] = [doy_to_datetime(y, d) for y, d in zip(df["Year"], df["Decimal_DOY"])]
    df["DOY"] = df["Decimal_DOY"].astype(int)
    return df


def load_cnr1raw_data(filepath):
    cols = [
        "Year", "Decimal_DOY", "HHMM",
        "SW_in", "SW_out", "LW_in_raw", "LW_out_raw", "T_body_K"
    ]
    df = pd.read_csv(filepath, header=None, names=cols, na_values=["NaN", "nan", "Inf", "-Inf"])
    df["Datetime"] = [doy_to_datetime(y, d) for y, d in zip(df["Year"], df["Decimal_DOY"])]
    df["DOY"] = df["Decimal_DOY"].astype(int)
    return df


def load_iws_data(filepath):
    df = pd.read_csv(filepath, header=None, na_values=["NaN", "nan", "Inf", "-Inf"])
    df = df.rename(columns={
        0: "Year", 1: "Month", 2: "Day", 3: "Hour", 4: "Minute",
        5: "Decimal_DOY", 19: "T_IWS", 20: "RH_IWS",
        22: "WS_IWS", 24: "WS_max_IWS", 25: "WD_IWS", 39: "P_hPa"
    })
    df["Datetime"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
    df["DOY"] = df["Decimal_DOY"].astype(int)
    return df

def audit_all_files_in_folder(folder_path):
    records = []
    rh_cols = [f"RH_{h}m" for h in ["8.0", "4.0", "2.0", "1.0", "0.5"]]
    t_cols = [f"T_{h}m" for h in ["8.0", "4.0", "2.0", "1.0", "0.5"]]
    ws_cols = [f"WS_{h}m" for h in ["8.0", "4.0", "2.0", "1.0", "0.5"]]
    wd_cols = [f"WD_{h}m" for h in ["8.0", "4.0", "2.0", "1.0", "0.5"]]
    prof_sensor_cols = rh_cols + t_cols + ws_cols + wd_cols

    for fname in ["profile.txt", "profile1min.txt"]:
        fpath = os.path.join(folder_path, fname)
        if os.path.exists(fpath):
            df = load_profile_data(fpath)
            n_total = len(df)
            n_nan = int(df[prof_sensor_cols].isna().any(axis=1).sum())
            n_neg = int((
                                (df[rh_cols] < 0).any(axis=1) |
                                (df[ws_cols] < 0).any(axis=1) |
                                (df[wd_cols] < 0).any(axis=1) |
                                (df[t_cols] < -40).any(axis=1)
                        ).sum())
            n_rh_over100 = int((df[rh_cols] > 100.0).any(axis=1).sum())
            records.append({
                "File": fname,
                "Total_Rows": n_total,
                "Start_Datetime": str(df["Datetime"].min()),
                "End_Datetime": str(df["Datetime"].max()),
                "DOY_Range": f"{df['DOY'].min()}–{df['DOY'].max()}",
                "Rows_with_NaN": n_nan,
                "Pct_NaN (%)": round(100.0 * n_nan / n_total, 2),
                "Rows_Negative_or_Corrupt": n_neg,
                "Notes": f"RH > 100% in {n_rh_over100} rows"
            })

    fpath_cnr1 = os.path.join(folder_path, "cnr1.txt")
    if os.path.exists(fpath_cnr1):
        df = load_cnr1_data(fpath_cnr1)
        n_total = len(df)
        rad_cols = ["SW_in", "SW_out", "LW_in", "LW_out", "R_net"]
        n_nan = int(df[rad_cols].isna().any(axis=1).sum())
        n_corrupt = int(((df["SW_in"] < -10.0) | (df["LW_in"] < 0) | (df["LW_out"] < 0)).sum())
        n_night_offset = int(((df["SW_in"] < 0.0) & (df["SW_in"] >= -10.0)).sum())
        records.append({
            "File": "cnr1.txt",
            "Total_Rows": n_total,
            "Start_Datetime": str(df["Datetime"].min()),
            "End_Datetime": str(df["Datetime"].max()),
            "DOY_Range": f"{df['DOY'].min()}–{df['DOY'].max()}",
            "Rows_with_NaN": n_nan,
            "Pct_NaN (%)": round(100.0 * n_nan / n_total, 2),
            "Rows_Negative_or_Corrupt": n_corrupt,
            "Notes": f"{n_night_offset} nighttime rows have small SW_in offset (-3 to 0 W/m2)"
        })

    fpath_cnr1raw = os.path.join(folder_path, "cnr1raw.txt")
    if os.path.exists(fpath_cnr1raw):
        df = load_cnr1raw_data(fpath_cnr1raw)
        n_total = len(df)
        rad_cols = ["SW_in", "SW_out", "LW_in_raw", "LW_out_raw", "T_body_K"]
        n_nan = int(df[rad_cols].isna().any(axis=1).sum())
        n_corrupt = int((df["SW_in"] < -10.0).sum())
        n_night_offset = int(((df["SW_in"] < 0.0) & (df["SW_in"] >= -10.0)).sum())
        records.append({
            "File": "cnr1raw.txt",
            "Total_Rows": n_total,
            "Start_Datetime": str(df["Datetime"].min()),
            "End_Datetime": str(df["Datetime"].max()),
            "DOY_Range": f"{df['DOY'].min()}–{df['DOY'].max()}",
            "Rows_with_NaN": n_nan,
            "Pct_NaN (%)": round(100.0 * n_nan / n_total, 2),
            "Rows_Negative_or_Corrupt": n_corrupt,
            "Notes": f"{n_night_offset} nighttime rows have small SW_in offset"
        })

    fpath_iws = os.path.join(folder_path, "iws.txt")
    if os.path.exists(fpath_iws):
        df = load_iws_data(fpath_iws)
        n_total = len(df)
        iws_cols = ["T_IWS", "RH_IWS", "WS_IWS", "WD_IWS", "P_hPa"]
        n_nan = int(df[iws_cols].isna().any(axis=1).sum())
        n_corrupt = int(((df["WS_IWS"] > 100.0) | (df["WS_IWS"] < 0) | (df["RH_IWS"] < 0)).sum())
        records.append({
            "File": "iws.txt",
            "Total_Rows": n_total,
            "Start_Datetime": str(df["Datetime"].min()),
            "End_Datetime": str(df["Datetime"].max()),
            "DOY_Range": f"{df['DOY'].min()}–{df['DOY'].max()}",
            "Rows_with_NaN": n_nan,
            "Pct_NaN (%)": round(100.0 * n_nan / n_total, 2),
            "Rows_Negative_or_Corrupt": n_corrupt,
            "Notes": "Checked for WS overflow (>100 m/s) and negative RH/WS"
        })

    return pd.DataFrame(records)

# FILTERING
def filter_profile_with_table(df, ws_limit=2.0):
    rh_cols = [f"RH_{h}m" for h in ["8.0", "4.0", "2.0", "1.0", "0.5"]]
    t_cols = [f"T_{h}m" for h in ["8.0", "4.0", "2.0", "1.0", "0.5"]]
    ws_cols = [f"WS_{h}m" for h in ["8.0", "4.0", "2.0", "1.0", "0.5"]]
    wd_cols = [f"WD_{h}m" for h in ["8.0", "4.0", "2.0", "1.0", "0.5"]]
    sensor_cols = rh_cols + t_cols + ws_cols + wd_cols

    n_init = len(df)
    steps = []

    steps.append({
        "Step": "0. Initial Raw Data",
        "Rows_Before": n_init,
        "Rows_Lost": 0,
        "Rows_Remaining": n_init,
        "Pct_Lost_of_Initial (%)": 0.0,
        "Cumulative_Pct_Lost (%)": 0.0
    })

    nan_mask = df[sensor_cols].isna().any(axis=1)
    df1 = df[~nan_mask].copy()
    lost_nan = int(nan_mask.sum())
    steps.append({
        "Step": "1. Missing (NaN) Sensor Values",
        "Rows_Before": n_init,
        "Rows_Lost": lost_nan,
        "Rows_Remaining": len(df1),
        "Pct_Lost_of_Initial (%)": round(100.0 * lost_nan / n_init, 2),
        "Cumulative_Pct_Lost (%)": round(100.0 * (n_init - len(df1)) / n_init, 2)
    })

    neg_mask = (
            (df1[rh_cols] < 0).any(axis=1) |
            (df1[ws_cols] < 0).any(axis=1) |
            (df1[wd_cols] < 0).any(axis=1) |
            (df1[wd_cols] > 360).any(axis=1) |
            (df1[t_cols] < -40).any(axis=1)
    )
    df2 = df1[~neg_mask].copy()
    df2[rh_cols] = df2[rh_cols].clip(upper=100.0)
    lost_neg = int(neg_mask.sum())
    steps.append({
        "Step": "2. Negative / Unphysical Values",
        "Rows_Before": len(df1),
        "Rows_Lost": lost_neg,
        "Rows_Remaining": len(df2),
        "Pct_Lost_of_Initial (%)": round(100.0 * lost_neg / n_init, 2),
        "Cumulative_Pct_Lost (%)": round(100.0 * (n_init - len(df2)) / n_init, 2)
    })

    north_mask = (df2["WD_8.0m"] >= NORTH_SECTOR_DEG[0]) | (df2["WD_8.0m"] <= NORTH_SECTOR_DEG[1])
    df3 = df2[~north_mask].copy()
    lost_north = int(north_mask.sum())
    steps.append({
        "Step": f"3. North Wind Sector ({NORTH_SECTOR_DEG[0]:.0f}°–{NORTH_SECTOR_DEG[1]:.0f}°)",
        "Rows_Before": len(df2),
        "Rows_Lost": lost_north,
        "Rows_Remaining": len(df3),
        "Pct_Lost_of_Initial (%)": round(100.0 * lost_north / n_init, 2),
        "Cumulative_Pct_Lost (%)": round(100.0 * (n_init - len(df3)) / n_init, 2)
    })

    ws_mask = (df3[ws_cols] > ws_limit).all(axis=1)
    df_final = df3[ws_mask].copy()
    lost_ws = int((~ws_mask).sum())
    steps.append({
        "Step": f"4. Low Wind Speed (All heights <= {ws_limit:.1f} m/s)",
        "Rows_Before": len(df3),
        "Rows_Lost": lost_ws,
        "Rows_Remaining": len(df_final),
        "Pct_Lost_of_Initial (%)": round(100.0 * lost_ws / n_init, 2),
        "Cumulative_Pct_Lost (%)": round(100.0 * (n_init - len(df_final)) / n_init, 2)
    })

    return df_final, pd.DataFrame(steps), df3

# SCRIPT
def main():
    print("=" * 80)
    print(f"PROPOSAL DATA ANALYSIS & STATISTICS — FOLDER: {FOLDER_PATH}")
    print("=" * 80)

    df_audit = audit_all_files_in_folder(FOLDER_PATH)
    df_audit.to_csv(os.path.join(OUTPUT_DIR, "stats_1_file_dates_and_errors.csv"), index=False)
    print("\n[TABLE 1] FILE DATES, TOTAL ROWS & WRONG DATA AUDIT:")
    print(df_audit.to_string(index=False))

    df_prof = load_profile_data(os.path.join(FOLDER_PATH, "profile.txt"))
    df_cnr1 = load_cnr1_data(os.path.join(FOLDER_PATH, "cnr1.txt"))

    df_2ms, tbl_2ms, df_pre_ws = filter_profile_with_table(df_prof, ws_limit=WS_PRIMARY_LIMIT)
    df_1ms, tbl_1ms, _ = filter_profile_with_table(df_prof, ws_limit=WS_FALLBACK_LIMIT)

    print(f"\n[TABLE 2A] FILTERING BREAKDOWN (Strict Proposal Criterion: WS > {WS_PRIMARY_LIMIT} m/s):")
    print(tbl_2ms.to_string(index=False))

    print(f"\n[TABLE 2B] FILTERING BREAKDOWN (Anemometer Threshold Criterion: WS > {WS_FALLBACK_LIMIT} m/s):")
    print(tbl_1ms.to_string(index=False))

    tbl_2ms["Criterion"] = "WS > 2.0 m/s"
    tbl_1ms["Criterion"] = "WS > 1.0 m/s"
    pd.concat([tbl_2ms, tbl_1ms], ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, "stats_2_filtering_breakdown.csv"), index=False
    )

    doy_summary = pd.DataFrame({
        "Raw_Rows": df_prof.groupby("DOY").size(),
        "After_NaN_Neg_North": df_pre_ws.groupby("DOY").size(),
        "Kept_WS>2.0m/s": df_2ms.groupby("DOY").size(),
        "Kept_WS>1.0m/s": df_1ms.groupby("DOY").size()
    }).fillna(0).astype(int)
    doy_summary["Date"] = [
        str(doy_to_datetime(2026, doy).date()) for doy in doy_summary.index
    ]
    doy_summary = doy_summary[["Date", "Raw_Rows", "After_NaN_Neg_North", "Kept_WS>2.0m/s", "Kept_WS>1.0m/s"]]
    doy_summary.to_csv(os.path.join(OUTPUT_DIR, "stats_3_daily_doy_counts.csv"))
    print("\n[TABLE 3] DATA POINTS PER DAY (DOY):")
    print(doy_summary.to_string())

    Active_DOYs = doy_summary[doy_summary["After_NaN_Neg_North"] > 0]
    if (Active_DOYs["Kept_WS>2.0m/s"] < MIN_ROWS_PER_DAY).any():
        print(f"\n  -> Using WS > {WS_FALLBACK_LIMIT} m/s so DOY 259 & 264 retain sufficient data.")
        df_clean = df_1ms.copy()
        ws_label = "WS > 1.0 m/s"
    else:
        df_clean = df_2ms.copy()
        ws_label = "WS > 2.0 m/s"

    ws_cols = [f"WS_{h}m" for h in ["8.0", "4.0", "2.0", "1.0", "0.5"]]
    t_cols = [f"T_{h}m" for h in ["8.0", "4.0", "2.0", "1.0", "0.5"]]
    rh_cols = [f"RH_{h}m" for h in ["8.0", "4.0", "2.0", "1.0", "0.5"]]
    wd_cols = [f"WD_{h}m" for h in ["8.0", "4.0", "2.0", "1.0", "0.5"]]

    desc_stats = df_clean[ws_cols + t_cols + rh_cols + wd_cols].describe().T[
        ["count", "mean", "std", "min", "50%", "max"]
    ].round(3)
    desc_stats.to_csv(os.path.join(OUTPUT_DIR, "stats_5_sensor_descriptive_stats.csv"))
    print(f"\n[TABLE 5] DESCRIPTIVE SENSOR STATISTICS ({ws_label}, n={len(df_clean)}):")
    print(desc_stats.to_string())

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for h in ["8.0", "4.0", "2.0", "1.0", "0.5"]:
        axes[0].plot(df_clean["Datetime"], df_clean[f"WS_{h}m"], label=f"{h} m", alpha=0.85)
        axes[1].plot(df_clean["Datetime"], df_clean[f"T_{h}m"], label=f"{h} m", alpha=0.85)
        axes[2].scatter(df_clean["Datetime"], df_clean[f"WD_{h}m"], label=f"{h} m", s=8, alpha=0.7)

    axes[0].set_ylabel("Wind Speed (m/s)")
    axes[0].set_title(f"Daily Evolution of Wind Speed, Temperature & Direction ({ws_label})")
    axes[0].legend(loc="upper right", ncol=5)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Temperature (°C)")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Wind Direction (°)")
    axes[2].set_xlabel("Date / Time")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_daily_evolution.png"), dpi=300)
    plt.show()

    ln_z = np.log(HEIGHTS)
    z_smooth = np.linspace(0.5, 8.0, 100)
    fig, ax = plt.subplots(figsize=(8, 8))

    fit_records = []

    for doy, grp in df_clean.groupby("DOY"):
        mean_ws = grp[ws_cols].mean().values
        slope, intercept, r_val, _, _ = linregress(ln_z, mean_ws)
        u_star = slope * KAPPA
        z0 = np.exp(-intercept / slope) if slope > 0 else np.nan

        mean_t = grp[t_cols].mean().values
        sl_t, _, r_t, _, _ = linregress(HEIGHTS, mean_t)

        fit_records.append({
            "Category": "Daily DOY",
            "Group": f"DOY {doy}",
            "N_Rows": len(grp),
            "WS_Log_R2": round(r_val ** 2, 4),
            "u_star (m/s)": round(u_star, 4),
            "z0 (m)": round(z0, 4),
            "T_Lin_R2": round(r_t ** 2, 4),
            "dT_dz (°C/m)": round(sl_t, 4)
        })

        pts = ax.plot(mean_ws, HEIGHTS, marker="o", linestyle="", alpha=0.7)
        c = pts[0].get_color()
        ax.plot(slope * np.log(z_smooth) + intercept, z_smooth, color=c,
                label=f"DOY {doy} ($R^2$={r_val ** 2:.3f}, $z_0$={z0:.3f}m)")

    mean_ws_all = df_clean[ws_cols].mean().values
    sl_all, int_all, r_all, _, _ = linregress(ln_z, mean_ws_all)
    u_star_all = sl_all * KAPPA
    z0_all = np.exp(-int_all / sl_all)
    sl_t_all, _, r_t_all, _, _ = linregress(HEIGHTS, df_clean[t_cols].mean().values)

    fit_records.append({
        "Category": "Overall",
        "Group": "Overall Mean",
        "N_Rows": len(df_clean),
        "WS_Log_R2": round(r_all ** 2, 4),
        "u_star (m/s)": round(u_star_all, 4),
        "z0 (m)": round(z0_all, 4),
        "T_Lin_R2": round(r_t_all ** 2, 4),
        "dT_dz (°C/m)": round(sl_t_all, 4)
    })

    ax.scatter(mean_ws_all, HEIGHTS, color="k", marker="s", s=60, zorder=5, label="Mean Data")
    ax.plot(sl_all * np.log(z_smooth) + int_all, z_smooth, "k--", lw=2.5,
            label=f"Mean Fit ($R^2$={r_all ** 2:.3f}, $z_0$={z0_all:.3f}m)")

    ax.set_yscale("log")
    ax.set_yticks(HEIGHTS)
    ax.set_yticks([], minor=True)
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Height (m) - Log Scale")
    ax.set_title("Wind Speed Log-Profile Fits")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_wind_log_profile_fits.png"), dpi=300)
    plt.show()

    # ==========================================================================
    # FIGURE 2B: DAILY & MEAN LOGARITHMIC TEMPERATURE PROFILE FITS
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(8, 8))

    for doy, grp in df_clean.groupby("DOY"):
        mean_t = grp[t_cols].mean().values
        # Logarithmic fit for temperature: T(z) = (theta_* / kappa) * ln(z / z0t) + T_ref
        slope_t, intercept_t, r_val_t, _, _ = linregress(ln_z, mean_t)

        pts = ax.plot(mean_t, HEIGHTS, marker="o", linestyle="", alpha=0.7)
        c = pts[0].get_color()
        ax.plot(slope_t * np.log(z_smooth) + intercept_t, z_smooth, color=c,
                label=f"DOY {doy} ($R^2$={r_val_t ** 2:.3f})")

    mean_t_all = df_clean[t_cols].mean().values
    sl_t_all, int_t_all, r_all_t, _, _ = linregress(ln_z, mean_t_all)

    ax.scatter(mean_t_all, HEIGHTS, color="k", marker="s", s=60, zorder=5, label="Mean Data")
    ax.plot(sl_t_all * np.log(z_smooth) + int_t_all, z_smooth, "k--", lw=2.5,
            label=f"Mean Fit ($R^2$={r_all_t ** 2:.3f})")

    ax.set_yscale("log")
    ax.set_yticks(HEIGHTS)
    ax.set_yticks([], minor=True)
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Height (m) - Log Scale")
    ax.set_title("Temperature Log-Profile Fits")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2b_temp_log_profile_fits.png"), dpi=300)
    plt.show()

    df_clean["dT_8_05"] = df_clean["T_8.0m"] - df_clean["T_0.5m"]
    df_clean["Stability"] = np.select(
        [df_clean["dT_8_05"] < -DT_NEUTRAL_BAND, df_clean["dT_8_05"] > DT_NEUTRAL_BAND],
        ["Unstable", "Stable"],
        default="Neutral"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    colors = {"Unstable": "tab:red", "Neutral": "tab:green", "Stable": "tab:blue"}

    for state in ["Unstable", "Neutral", "Stable"]:
        sub = df_clean[df_clean["Stability"] == state]
        if len(sub) == 0:
            continue

        ws_prof = sub[ws_cols].mean().values
        sl_w, int_w, r_w, _, _ = linregress(ln_z, ws_prof)
        u_st = sl_w * KAPPA
        z0_s = np.exp(-int_w / sl_w) if sl_w > 0 else np.nan

        t_prof = (sub[t_cols].sub(sub["T_0.5m"], axis=0)).mean().values
        sl_t, int_t, r_t, _, _ = linregress(HEIGHTS, t_prof)

        fit_records.append({
            "Category": "Stability (dT)",
            "Group": state,
            "N_Rows": len(sub),
            "WS_Log_R2": round(r_w ** 2, 4),
            "u_star (m/s)": round(u_st, 4),
            "z0 (m)": round(z0_s, 4),
            "T_Lin_R2": round(r_t ** 2, 4),
            "dT_dz (°C/m)": round(sl_t, 4)
        })

        axes[0].scatter(ws_prof, HEIGHTS, color=colors[state], s=50, zorder=5)
        axes[0].plot(sl_w * np.log(z_smooth) + int_w, z_smooth, "-", color=colors[state], lw=2,
                     label=f"{state} (n={len(sub)}, $R^2$={r_w ** 2:.3f}, $z_0$={z0_s:.3f}m)")

        axes[1].scatter(t_prof, HEIGHTS, color=colors[state], marker="s", s=50, zorder=5)
        axes[1].plot(sl_t * z_smooth + int_t, z_smooth, "-", color=colors[state], lw=2,
                     label=f"{state} ($dT/dz$={sl_t:+.3f} °C/m, $R^2$={r_t ** 2:.3f})")

    axes[0].set_yscale("log")
    axes[0].set_yticks(HEIGHTS)
    axes[0].set_yticks([], minor=True)
    axes[0].get_yaxis().set_major_formatter(plt.ScalarFormatter())
    axes[0].set_xlabel("Mean Wind Speed (m/s)")
    axes[0].set_ylabel("Height (m) - Log Scale")
    axes[0].set_title("Wind Profiles by Stability")
    axes[0].grid(True, ls="--", alpha=0.4)
    axes[0].legend(loc="lower right", fontsize=9)

    axes[1].axvline(0, color="k", ls=":", alpha=0.5)
    axes[1].set_yticks(HEIGHTS, [str(h) for h in HEIGHTS])
    axes[1].set_xlabel(r"Temperature Difference $T(z) - T(0.5\mathrm{m})$ (°C)")
    axes[1].set_ylabel("Height (m) - Linear Scale")
    axes[1].set_title("Temperature Profiles by Stability")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_stability_profiles.png"), dpi=300)
    plt.show()

    def get_sector(wd):
        if 45 < wd <= 135:
            return "East (45°–135°)"
        elif 135 < wd <= 225:
            return "South (135°–225°)"
        elif 225 < wd <= 270:
            return "SW–West (225°–270°)"
        else:
            return "WNW (270°–315°)"

    df_clean["Sector"] = df_clean["WD_8.0m"].apply(get_sector)
    fig, ax = plt.subplots(figsize=(8, 7))

    for sec, grp in df_clean.groupby("Sector"):
        if len(grp) < 3:
            continue
        ws_sec = grp[ws_cols].mean().values
        sl_s, int_s, r_s, _, _ = linregress(ln_z, ws_sec)
        u_st_s = sl_s * KAPPA
        z0_sec = np.exp(-int_s / sl_s) if sl_s > 0 else np.nan
        sl_t_s, _, r_t_s, _, _ = linregress(HEIGHTS, grp[t_cols].mean().values)

        fit_records.append({
            "Category": "Fetch Sector",
            "Group": sec,
            "N_Rows": len(grp),
            "WS_Log_R2": round(r_s ** 2, 4),
            "u_star (m/s)": round(u_st_s, 4),
            "z0 (m)": round(z0_sec, 4),
            "T_Lin_R2": round(r_t_s ** 2, 4),
            "dT_dz (°C/m)": round(sl_t_s, 4)
        })

        pts = ax.plot(ws_sec, HEIGHTS, marker="o", linestyle="", markersize=7)
        c = pts[0].get_color()
        ax.plot(sl_s * np.log(z_smooth) + int_s, z_smooth, lw=2, color=c,
                label=f"{sec} (n={len(grp)}, $R^2$={r_s ** 2:.3f}, $z_0$={z0_sec:.3f}m)")

    ax.set_yscale("log")
    ax.set_yticks(HEIGHTS)
    ax.set_yticks([], minor=True)
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Mean Wind Speed (m/s)")
    ax.set_ylabel("Height (m) - Log Scale")
    ax.set_title("Wind Speed Log-Profiles by Fetch Sector")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_fetch_sector_profiles.png"), dpi=300)
    plt.show()

    df_fits = pd.DataFrame(fit_records)
    df_fits.to_csv(os.path.join(OUTPUT_DIR, "stats_4_profile_fits_summary.csv"), index=False)
    print("\n[TABLE 4] SUMMARY OF ALL LOG-WIND & LINEAR-TEMPERATURE PROFILE FITS:")
    print(df_fits.to_string(index=False))

    df_cnr1_clean = df_cnr1[~(df_cnr1["R_net"].isna() | (df_cnr1["SW_in"] < -10.0))].copy()

    df_clean["Datetime_min"] = df_clean["Datetime"].dt.round("1min")
    df_cnr1_clean["Datetime_min"] = df_cnr1_clean["Datetime"].dt.round("1min")

    df_flux = pd.merge(
        df_clean,
        df_cnr1_clean[["Datetime_min", "R_net", "SW_in", "LW_net"]],
        on="Datetime_min",
        how="inner"
    )

    for h in ["8.0", "0.5"]:
        es = 0.61078 * np.exp((17.27 * df_flux[f"T_{h}m"]) / (df_flux[f"T_{h}m"] + 237.3))
        e_kpa = es * (df_flux[f"RH_{h}m"] / 100.0)
        df_flux[f"q_{h}m"] = 0.622 * e_kpa / 101.3

    d_theta = (df_flux["T_8.0m"] - df_flux["T_0.5m"]) + GAMMA_D * (8.0 - 0.5)
    d_q = df_flux["q_8.0m"] - df_flux["q_0.5m"]
    d_u = df_flux["WS_8.0m"] - df_flux["WS_0.5m"]
    ln_z2_z1 = np.log(8.0 / 0.5)

    df_flux["u_star"] = KAPPA * d_u / ln_z2_z1
    df_flux["theta_star"] = KAPPA * d_theta / ln_z2_z1
    df_flux["q_star"] = KAPPA * d_q / ln_z2_z1

    df_flux["H_profile"] = -RHO_AIR * CP_AIR * df_flux["u_star"] * df_flux["theta_star"]
    df_flux["LE_profile"] = -RHO_AIR * LV_WATER * df_flux["u_star"] * df_flux["q_star"]

    plt.figure(figsize=(12, 6))
    plt.plot(df_flux["Datetime"], df_flux["R_net"], "k-", lw=1.8, label=r"Net Radiation ($Q^*$)")
    plt.plot(df_flux["Datetime"], df_flux["H_profile"], "r-", lw=1.5,
             label=r"Sensible Heat Flux ($H_{\mathrm{profile}}$)")
    plt.plot(df_flux["Datetime"], df_flux["LE_profile"], "b-", lw=1.5,
             label=r"Latent Heat Flux ($LE_{\mathrm{profile}}$)")
    plt.axhline(0, color="gray", ls=":")
    plt.ylabel(r"Flux ($\mathrm{W\ m^{-2}}$)")
    plt.xlabel("Date / Time")
    plt.title("Aerodynamic Profile Fluxes ($H$, $LE$) & Net Radiation ($Q^*$)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_profile_fluxes.png"), dpi=300)
    plt.show()

    df_flux.to_csv(os.path.join(OUTPUT_DIR, "cleaned_proposal_mast_data.csv"), index=False)
    print(f"\n[DONE] Saved 5 figures and 6 CSV statistical tables to '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()