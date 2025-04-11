from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
from pprint import pprint
import matplotlib.pyplot as plt


def plot_aggregated_lab_boxplots(agg_dict, safety_threshold=5):
    """
    Plots box plots for lab variables using only the aggregated results.

    The input `agg_dict` should be the dictionary corresponding to
    "Laboratory Values (Aggregated)" from our flexible_stats output. Each key in the
    dictionary (for lab variables) should contain a dictionary with:
      - "mean": Aggregated mean value
      - "std": Aggregated standard deviation
      - "median": Aggregated median value
      - "25%": The 25th percentile value
      - "75%": The 75th percentile value

    For each variable, the function computes:
      - IQR = Q3 - Q1
      - Lower whisker = Q1 - 1.5 * IQR
      - Upper whisker = Q3 + 1.5 * IQR
      - The mean is also shown as a separate marker.

    Parameters:
      agg_dict (dict): Aggregated lab values dictionary (from "Laboratory Values (Aggregated)")
      safety_threshold (int): Minimum number of groups required. If the number of groups
                              is below this threshold, a warning annotation is added to the plot.

    Returns:
      fig (matplotlib.figure.Figure): The generated figure containing the box plots.
    """
    boxplot_data = []
    labels = []
    
    for var, stats in agg_dict.items():
        # Skip entries that are not dictionaries (e.g., total_patients)
        if not isinstance(stats, dict):
            continue
        if all(k in stats for k in ["Q1", "median", "Q3", "mean"]):
            q1 = stats["Q1"]
            med = stats["median"]
            q3 = stats["Q3"]
            IQR = q3 - q1
            whislo = q1 - 1.5 * IQR
            whishi = q3 + 1.5 * IQR
            box_dict = {
                "label": var,
                "whislo": round(whislo, 2),
                "q1": round(q1, 2),
                "med": round(med, 2),
                "q3": round(q3, 2),
                "whishi": round(whishi, 2),
                "mean": round(stats["mean"], 2),
                "fliers": []
            }
            boxplot_data.append(box_dict)
            labels.append(var)
    
    # Create a figure and axis.
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # If no valid data is found, annotate a warning and return the figure.
    if not boxplot_data:
        ax.text(0.5, 0.5, "No valid aggregated lab values available", transform=ax.transAxes,
                color="red", fontsize=14, ha="center")
        ax.set_title("Aggregated Laboratory Values (Per-Patient)")
        plt.tight_layout()
        return fig

    bp = ax.bxp(boxplot_data, showmeans=True, patch_artist=True)
    ax.set_title("Aggregated Laboratory Values (Per-Patient)")
    ax.set_ylabel("Value")
    
    # Safety threshold check: if total_patients is available in agg_dict, annotate warning if needed.
    num_patients = agg_dict.get("total_patients", None)
    if num_patients is not None and num_patients < safety_threshold:
        ax.text(0.5, 0.95, f"Warning: Only {num_patients} groups (<{safety_threshold}) available.",
                transform=ax.transAxes, color="red", fontsize=12, ha="center")
    
    plt.tight_layout()
    return fig

dataset_1 = {'database': './tests/test_strata.csv', 'db_type': 'csv'}
datasets = [[dataset_1],]
org_ids = ids = [0,]

client = MockAlgorithmClient(
    datasets = datasets,
    organization_ids=org_ids,
    module='v6_strata_fit_stats_py',
    collaboration_id=None,
    node_ids=None
)

organizations = client.organization.list()
org_ids = ids = [organization["id"] for organization in organizations]

average_task = client.task.create(
    input_={
        'method': 'partial_stats',
        'kwargs': {}
    },
    organizations=[org_ids[0]]
)

results = client.result.get(average_task.get("id"))
pprint(results)

# Extract the aggregated lab values dictionary.
agg_lab_values = results.get("laboratory_values_grouped_by_pat_id", {})

# Optionally, if you need the total_patients for the safety threshold:
if "total_patients" not in agg_lab_values:
    total_patients = results.get("Visits Per Time Period", {}).get("total_patients", None)
    if total_patients is not None:
        agg_lab_values["total_patients"] = total_patients


fig = plot_aggregated_lab_boxplots(agg_lab_values, safety_threshold=5)
plt.show()