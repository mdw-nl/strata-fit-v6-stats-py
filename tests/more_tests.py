from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
import pandas as pd
from pprint import pprint

# Dataset is our simple CSV in tests/test_data.csv.
dataset_1 = {'database': './tests/test_strata.csv', 'db_type': 'csv'}
datasets = [[dataset_1],]
org_ids = [0,]

client = MockAlgorithmClient(
    datasets=datasets,
    organization_ids=org_ids,
    module='v6_basic_stats_py',
    collaboration_id=None,
    node_ids=None
)

# Get organization ids from the client.
organizations = client.organization.list()
org_ids = [org["id"] for org in organizations]

# -------------------------
# Task 1: Generic Statistics (Non-Grouped)
# -------------------------
generic_task = client.task.create(
    input_={
        'method': 'flexible_stats',
        'kwargs': {
            'numeric_stats': ["count", "mean", "std", "median"],
            'categorical_stats': ["count", "unique"]
        }
    },
    organizations=[org_ids[0]]
)
generic_results = client.result.get(generic_task.get("id"))
print("Generic Stats:")
pprint(generic_results)

# -------------------------
# Task 2: Visit Intensity
# Group by 'pat_id' to simulate counting visits per patient and then aggregate these counts.
# -------------------------
visit_intensity_task = client.task.create(
    input_={
        'method': 'flexible_stats',
        'kwargs': {
            'numeric_stats': ["count"],
            'group_by': 'pat_ID'
        }
    },
    organizations=[org_ids[0]]
)
visit_intensity_results = client.result.get(visit_intensity_task.get("id"))
print("Visit Intensity Stats:")
pprint(visit_intensity_results)

# -------------------------
# Task 3: Demographics
# Simulate extraction of demographic data (e.g. Age_diagnosis, Sex)
# For testing purposes, 'value' serves as a stand-in for age and 'category' for sex.
# -------------------------
demographics_task = client.task.create(
    input_={
        'method': 'flexible_stats',
        'kwargs': {
            'numeric_stats': ["mean", "std", "median"],
            'categorical_stats': ["count", "unique"]
        }
    },
    organizations=[org_ids[0]]
)
demographics_results = client.result.get(demographics_task.get("id"))
print("Demographics Stats:")
pprint(demographics_results)

# -------------------------
# Task 4: Calendar Year Distribution / DMARD Changes per Year (Simulated)
# In a real dataset these would come from dedicated columns (e.g. Year_diagnosis, N_prev_csDMARD).
# Here we simulate by grouping on 'pat_id' and extracting numeric stats from 'value'.
# -------------------------
year_dmard_task = client.task.create(
    input_={
        'method': 'flexible_stats',
        'kwargs': {
            'numeric_stats': ["mean", "std", "median"],
            'group_by': 'pat_ID'
        }
    },
    organizations=[org_ids[0]]
)
year_dmard_results = client.result.get(year_dmard_task.get("id"))
print("Calendar Year/DMARD Changes Stats (Simulated):")
pprint(year_dmard_results)
