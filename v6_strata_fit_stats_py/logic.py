import pandas as pd
import numpy as np
from typing import Any, Dict

from .types import (
    enforce_output_schema,
    UniquePatientsOutput,
    VisitDefinitionOutput,
    VisitsPerTimePeriodOutput,
    MissingDataPerVisitOutput,
    DemographicsOutput,
    DiseaseDurationDistributionOutput,
    PartialStatsOutput
)

# Define a privacy threshold: any count below this value is suppressed.
PRIVACY_THRESHOLD = 5

@enforce_output_schema(UniquePatientsOutput)
def unique_patients(df: pd.DataFrame):
    """
    1. Unique Patients Per Center:
       Count the number of unique patient IDs.
       If the count is below the privacy threshold, report "<{threshold}".
    """
    count = int(df["pat_ID"].nunique())
    if count < PRIVACY_THRESHOLD:
        return f"<{PRIVACY_THRESHOLD}"
    return {"unique_patients": count}

@enforce_output_schema(VisitDefinitionOutput)
def check_visit_definition(df: pd.DataFrame):
    """
    2. Check Visit Definition:
       For each patient (grouped by pat_ID), sort visits by 'Visit_months_from_diagnosis'.
       For each visit (after the first), compare DMARD-related variables to the previous visit 
       and check if all disease activity variables are missing.
       
       Returns a dictionary with the total count of invalid visits.
       (Note: The behavior when values are None is left as a TODO for clinical input.)
    """
    dmard_cols = ["csDMARD1", "csDMARD2", "csDMARD3", "bDMARD", "tsDMARD", "GC"]
    disease_activity_cols = ["DAS28", "ESR", "CRP", "TJC28", "SJC28", "Pat_global", "Ph_global", "Pain"]
    invalid_count = 0

    for pat_id, group in df.groupby("pat_ID"):
        group = group.sort_values("Visit_months_from_diagnosis").reset_index(drop=True)
        for i in range(1, len(group)):
            current = group.iloc[i]
            previous = group.iloc[i - 1]
            # For DMARD columns, treat both values missing as "unchanged"
            dmard_unchanged = all(
                (pd.isna(current[col]) and pd.isna(previous[col])) or (current[col] == previous[col])
                for col in dmard_cols if col in group.columns
            )
            disease_missing = all(pd.isna(current[col]) or current[col] == ""
                                  for col in disease_activity_cols if col in group.columns)
            if dmard_unchanged and disease_missing:
                invalid_count += 1
    return {"invalid_visits": invalid_count}

@enforce_output_schema(VisitsPerTimePeriodOutput)
def visits_per_time_period(df: pd.DataFrame):
    """
    3. Visits Per Time Period:
       For each patient, calculate:
         - Total number of visits.
         - Total follow-up time (difference between last and first visit).
         - Visit rate = (number of visits) / (total follow-up time).
       
       Returns a dictionary of overall summary statistics (mean, std, median) of the visit rate,
       along with the total patient count.
    """
    results = []
    for _, group in df.groupby("pat_ID"):
        group = group.sort_values("Visit_months_from_diagnosis")
        visits_count = len(group)
        min_visit = group["Visit_months_from_diagnosis"].min()
        max_visit = group["Visit_months_from_diagnosis"].max()
        total_follow_up = max_visit - min_visit if visits_count > 1 else np.nan
        rate = visits_count / total_follow_up if total_follow_up and total_follow_up > 0 else np.nan
        results.append(rate)
    rates = pd.Series(results).dropna()
    overall_stats = {
        "visit_rate_mean": round(rates.mean(), 3) if not rates.empty else None,
        "visit_rate_std": round(rates.std(), 3) if not rates.empty else None,
        "visit_rate_median": round(rates.median(), 3) if not rates.empty else None,
        "total_patients": int(df["pat_ID"].nunique())
    }
    return overall_stats

@enforce_output_schema(MissingDataPerVisitOutput)
def missing_data_per_visit(df: pd.DataFrame):
    """
    4. Missing Data Per Feature:
       Check each visit to see if ALL key clinical/lab variables are missing.
       
       Returns a dictionary with:
         - Count of visits with all missing values.
         - Total number of visits.
         - Percentage of such visits.
    """
    cols = ["DAS28", "ESR", "CRP", "TJC28", "SJC28", "Pat_global", "Ph_global", "Pain"]
    all_missing = df[cols].apply(lambda row: all(pd.isna(x) or x == "" for x in row), axis=1)
    count_missing = all_missing.sum()
    total = len(df)
    percent_missing = round((count_missing / total) * 100, 2) if total > 0 else 0
    return {
        "count_all_missing": int(count_missing),
        "total_visits": total,
        "percent_all_missing": percent_missing
    }

def safe_counts_and_proportions_groupwise(counts: Dict[Any, int], threshold=PRIVACY_THRESHOLD):
    """
    Masks *all* counts and proportions if any group has a count below the threshold.

    Returns:
        safe_counts: Dict[Any, Union[int, str]]
        safe_proportions: Dict[Any, Union[float, str]]
    """
    if any(v < threshold for v in counts.values()):
        return (
            {k: f"<{threshold}" for k in counts},
            {k: "masked" for k in counts}
        )

    total = sum(counts.values())
    safe_counts = {k: v for k, v in counts.items()}
    safe_proportions = {k: round(v / total, 3) if total > 0 else None for k, v in counts.items()}
    return safe_counts, safe_proportions


@enforce_output_schema(DemographicsOutput)
def demographics_stats(df: pd.DataFrame):
    """
    5. Demographics:
       For continuous variables (e.g. Age_diagnosis), compute mean and std.
       For categorical variables (e.g. Sex, RF_positivity, anti_CCP), compute counts and proportions.
       Small counts (below threshold) are suppressed for privacy.
       
       Returns a dictionary of computed statistics.
    """
    results = {}
    # Continuous variable: Age_diagnosis
    age_column_name = "Age_diagnosis"
    if age_column_name in df.columns:
        results["Age_mean"] = round(df[age_column_name].mean(), 2)
        results["Age_std"] = round(df[age_column_name].std(), 2)
    else:
        raise KeyError(f'Column "{age_column_name}" is not present in the dataset - review the data schema adherence!')
    
    # Categorical variables
    for var in ["Sex", "RF_positivity", "anti_CCP"]:
        if var in df.columns:
            counts = df[var].value_counts(dropna=False).to_dict()
            safe_counts, safe_proportions = safe_counts_and_proportions_groupwise(counts)
            results[f"{var}_counts"] = safe_counts
            results[f"{var}_proportions"] = safe_proportions
        else:
            raise KeyError(f'Column "{var}" is not present in the dataset - review the data schema adherence!')
    
    return results

@enforce_output_schema(DiseaseDurationDistributionOutput)
def disease_duration_distribution(df: pd.DataFrame):
    """
    6. Disease Duration Distribution:
       Compute the distribution (mean, std, skewness) of the Year_diagnosis variable.
       Note: Deduplicates per patient before computing stats.
    """
    year_column = "Year_diagnosis"
    groupping_column = "pat_ID"
    if year_column in df.columns and groupping_column in df.columns:
        # Group by patient and take the first non-null value
        patient_level = (
            df.dropna(subset=[year_column])
              .groupby(groupping_column)[year_column]
              .first()
              .apply(pd.to_numeric, errors="coerce")
              .dropna()
        )
        return {
            f"{year_column}_mean": round(patient_level.mean(), 2),
            f"{year_column}_std": round(patient_level.std(), 2),
            f"{year_column}_skewness": round(patient_level.skew(), 2)
        }
    else:
        error_message = (
            f'Columns "{year_column}" and "{groupping_column}" are '
            'not present in the dataset - review the data schema adherence!'
        )
        raise KeyError(error_message)


def lab_values_stats_overall(df: pd.DataFrame):
    """
    7a. Laboratory Values (Overall):
       Compute overall descriptive statistics for lab variables.
    """
    lab_vars = ["CRP", "ESR", "TJC28", "SJC28", "DAS28", "Pat_global", "Ph_global", "Pain"]
    results = {}
    missing_columns = []
    for var in lab_vars:
        if var in df.columns:
            series = pd.to_numeric(df[var], errors='coerce')
            mean_val = series.mean()
            std_val = series.std()
            skewness_val = series.skew()
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)].tolist()
            results[var] = {
                "mean": round(mean_val, 2),
                "std": round(std_val, 2),
                "skewness": round(skewness_val, 2),
                "outlier_count": len(outliers)
            }
        else:
            missing_columns.append(var)
    if missing_columns:
        error_message = (
            f'Columns {missing_columns} are '
            'not present in the dataset - review the data schema adherence!'
        )
        raise KeyError(error_message)
    return results

def lab_values_stats_aggregated(df: pd.DataFrame):
    """
    7b. Laboratory Values (Aggregated):
       Aggregate lab values by patient and summarize those aggregates.
    """
    if "pat_ID" not in df.columns:
        raise ValueError("Grouping requested but 'pat_ID' column not found.")
    lab_vars = ["CRP", "ESR", "TJC28", "SJC28", "DAS28", "Pat_global", "Ph_global", "Pain"]
    missing_columns = []
    grouped = df.groupby("pat_ID")
    results = {}
    for var in lab_vars:
        if var in df.columns:
            means = grouped[var].mean().dropna()
            results[var] = {
                "mean": round(means.mean(), 2),
                "std": round(means.std(), 2),
                "median": round(means.median(), 2),
                "Q1": round(means.quantile(0.25), 2),
                "Q3": round(means.quantile(0.75), 2)
            }
        else:
            missing_columns.append(var)
    if missing_columns:
        error_message = (
            f'Columns {missing_columns} are '
            'not present in the dataset - review the data schema adherence!'
        )
        raise KeyError(error_message)
    return results

@enforce_output_schema(PartialStatsOutput)
def compute_partial_stats(df: pd.DataFrame):
    """
    Aggregates various statistics for the dataset while preserving privacy.
    Calls individual functions for:
      1. Unique Patients Per Center
      2. Check Visit Definition
      3. Visits Per Time Period
      4. Missing Data Per Visit
      5. Demographics
      6. Disease Duration Distribution
      7. Laboratory Values (Overall and Aggregated)
      
    Returns a dictionary with all computed results.
    """
    unique = unique_patients(df)
    visit_def = check_visit_definition(df)
    visits_overall = visits_per_time_period(df)
    missing_data = missing_data_per_visit(df)
    demographics = demographics_stats(df)
    duration = disease_duration_distribution(df)
    lab_overall = lab_values_stats_overall(df)
    lab_grouped = lab_values_stats_aggregated(df)

    results = {
        "unique_patients_per_center": unique,
        "check_visit_definition": visit_def,
        "visits_per_time_period": visits_overall,
        "missing_data_per_visit": missing_data,
        "demographics": demographics,
        "disease_duration_distribution": duration,
        "laboratory_values_overall": lab_overall,
        "laboratory_values_grouped_by_pat_id": lab_grouped
    }

    return results
