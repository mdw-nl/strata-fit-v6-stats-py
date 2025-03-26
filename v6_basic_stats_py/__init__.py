import pandas as pd
import numpy as np
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import data

def unique_patients(df: pd.DataFrame, id_column="pat_ID"):
    """
    1. Unique Patients Per Center:
       Count the number of unique patient IDs.
    """
    return df[id_column].nunique()

def check_visit_definition(df: pd.DataFrame):
    """
    2. Check Visit Definition:
       For each patient (grouped by pat_ID), first sort visits by 
       'Visit_months_from_diagnosis'. Then, for each visit (after the first),
       compare the current DMARD-related variables to the previous visit. 
       Also, check if all disease activity variables are missing.
       
       - DMARD-related columns: csDMARD1, csDMARD2, csDMARD3, bDMARD, tsDMARD, GC
       - Disease activity columns: DAS28, ESR, CRP, TJC28, SJC28, Pat_global, Ph_global, Pain
       
       A visit is flagged as invalid if there is no change in DMARD values 
       and all disease activity variables are missing. The function returns
       the total count of such invalid visits.
    """
    dmard_cols = ["csDMARD1", "csDMARD2", "csDMARD3", "bDMARD", "tsDMARD", "GC"]
    disease_activity_cols = ["DAS28", "ESR", "CRP", "TJC28", "SJC28", "Pat_global", "Ph_global", "Pain"]
    invalid_count = 0

    for pat_id, group in df.groupby("pat_ID"):
        # Sort visits in chronological order
        group = group.sort_values("Visit_months_from_diagnosis").reset_index(drop=True)
        for i in range(1, len(group)):
            current = group.iloc[i]
            previous = group.iloc[i - 1]
            # Compare DMARD-related columns (only those that exist in the data)
            dmard_unchanged = all(current[col] == previous[col] for col in dmard_cols if col in group.columns)
            # Check that all disease activity variables are missing (NaN or empty string)
            disease_missing = all(pd.isna(current[col]) or current[col] == "" for col in disease_activity_cols if col in group.columns)
            if dmard_unchanged and disease_missing:
                invalid_count += 1
    return invalid_count

def visits_per_time_period(df: pd.DataFrame):
    """
    3. Visits Per Time Period:
       For each patient, calculate:
         - Total number of visits.
         - Total follow-up time (difference between last and first visit, 
           using 'Visit_months_from_diagnosis').
         - Visit rate = (number of visits) / (total follow-up time).
       
       Returns a tuple:
         - A DataFrame with per-patient results.
         - A dictionary of overall descriptive statistics (mean and standard deviation) of the visit rate.
    """
    results = []
    for pat_id, group in df.groupby("pat_ID"):
        group = group.sort_values("Visit_months_from_diagnosis")
        visits_count = len(group)
        min_visit = group["Visit_months_from_diagnosis"].min()
        max_visit = group["Visit_months_from_diagnosis"].max()
        # Only compute follow-up if more than one visit is present
        total_follow_up = max_visit - min_visit if visits_count > 1 else np.nan
        rate = visits_count / total_follow_up if total_follow_up and total_follow_up > 0 else np.nan
        results.append({
            "pat_ID": pat_id, 
            "visits_count": visits_count, 
            "total_follow_up": total_follow_up, 
            "visit_rate": rate
        })
    result_df = pd.DataFrame(results)
    overall_stats = result_df["visit_rate"].agg(["mean", "std"]).to_dict()
    return result_df, overall_stats

def missing_data_per_visit(df: pd.DataFrame):
    """
    4. Missing Data Per Feature:
       Check each visit to see if ALL key clinical/lab variables are missing.
       The key variables include: DAS28, ESR, CRP, TJC28, SJC28, Pat_global, Ph_global, Pain.
       
       Returns a dictionary with:
         - The count of visits with all missing values.
         - Total number of visits.
         - The percentage of such visits.
    """
    cols = ["DAS28", "ESR", "CRP", "TJC28", "SJC28", "Pat_global", "Ph_global", "Pain"]
    df["all_missing"] = df[cols].apply(lambda row: all(pd.isna(x) or x == "" for x in row), axis=1)
    count_missing = df["all_missing"].sum()
    total = len(df)
    percent_missing = (count_missing / total) * 100 if total > 0 else 0
    return {"count_all_missing": int(count_missing), "total_visits": total, "percent_all_missing": percent_missing}

def demographics_stats(df: pd.DataFrame):
    """
    5. Demographics:
       For continuous variables (e.g. Age), compute mean and standard deviation.
       For categorical variables (e.g. Sex, RF_positivity, anti_CCP), compute counts
       and proportions.
       
       Returns a dictionary of computed statistics.
    """
    results = {}
    # For Age (or Age_diagnosis if available)
    if "Age" in df.columns:
        results["Age_mean"] = df["Age"].mean()
        results["Age_std"] = df["Age"].std()
    # For categorical variables
    for var in ["Sex", "RF_positivity", "anti_CCP"]:
        if var in df.columns:
            counts = df[var].value_counts(dropna=False)
            total = counts.sum()
            proportions = (counts / total).to_dict()
            results[f"{var}_counts"] = counts.to_dict()
            results[f"{var}_proportions"] = proportions
    return results

def disease_duration_distribution(df: pd.DataFrame):
    """
    6. Disease Duration Distribution:
       Compute the distribution of the diagnosis year (or a proxy for disease duration).
       Returns the mean, standard deviation, and skewness of the Year_diagnosis variable.
    """
    if "Year_diagnosis" in df.columns:
        diagnosis_year = pd.to_numeric(df["Year_diagnosis"], errors='coerce')
        return {
            "Year_diagnosis_mean": diagnosis_year.mean(),
            "Year_diagnosis_std": diagnosis_year.std(),
            "Year_diagnosis_skewness": diagnosis_year.skew()
        }
    else:
        return {}

def lab_values_stats(df: pd.DataFrame, group_by=False):
    """
    7. Laboratory Values:
       For lab variables (CRP, ESR, TJC28, SJC28, DAS28, Pat_global, Ph_global, Pain),
       compute descriptive statistics.
       
       If group_by is False:
         - Compute overall descriptive stats (mean, std, skewness).
         - Identify outliers using the IQR method.
         
       If group_by is True:
         - Group by pat_ID and aggregate the per-patient means (or other second-order statistics)
           to ensure individual visit details are not exposed.
       
       Returns a dictionary of statistics for each lab variable.
    """
    lab_vars = ["CRP", "ESR", "TJC28", "SJC28", "DAS28", "Pat_global", "Ph_global", "Pain"]
    results = {}
    if not group_by:
        for var in lab_vars:
            if var in df.columns:
                series = pd.to_numeric(df[var], errors='coerce')
                mean_val = series.mean()
                std_val = series.std()
                skewness_val = series.skew()
                # Identify outliers using IQR
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = series[(series < lower_bound) | (series > upper_bound)].tolist()
                results[var] = {
                    "mean": mean_val, 
                    "std": std_val, 
                    "skewness": skewness_val, 
                    "outliers": outliers
                }
    else:
        # When grouping, ensure we do not expose visit-level details
        if "pat_ID" not in df.columns:
            raise ValueError("Grouping requested but 'pat_ID' column not found.")
        grouped = df.groupby("pat_ID")
        for var in lab_vars:
            if var in df.columns:
                # Compute per-patient means, then aggregate those values
                means = grouped[var].mean().dropna()
                results[var] = {
                    "count": int(means.count()),
                    "mean": means.mean(),
                    "std": means.std(),
                    "median": means.median(),
                    "25%": means.quantile(0.25),
                    "75%": means.quantile(0.75)
                }
    return results

@data(1)
def flexible_stats(df: pd.DataFrame, **kwargs):
    """
    Aggregates various statistics for the dataset while preserving privacy.
    Calls individual functions for:
      1. Unique Patients Per Center
      2. Check Visit Definition
      3. Visits Per Time Period
      4. Missing Data Per Visit
      5. Demographics
      6. Disease Duration Distribution
      7. Laboratory Values (both overall and grouped by pat_ID)
      
    Returns a dictionary with all computed results.
    """
    unique = unique_patients(df)
    visit_def = check_visit_definition(df)
    visits_df, visits_overall = visits_per_time_period(df)
    missing_data = missing_data_per_visit(df)
    demo = demographics_stats(df)
    duration = disease_duration_distribution(df)
    lab_overall = lab_values_stats(df, group_by=False)
    lab_grouped = lab_values_stats(df, group_by=True)
    
    results = {
        "Unique Patients Per Center": unique,
        "Check Visit Definition (invalid visits count)": visit_def,
        "Visits Per Time Period": {
            "per_patient": visits_df.to_dict(orient="records"),
            "overall_stats": visits_overall
        },
        "Missing Data Per Visit": missing_data,
        "Demographics": demo,
        "Disease Duration Distribution": duration,
        "Laboratory Values (Overall)": lab_overall,
        "Laboratory Values (Grouped by pat_ID)": lab_grouped
    }
    
    info(results)
    return results
