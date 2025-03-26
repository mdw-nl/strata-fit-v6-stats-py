from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
import pandas as pd
from pprint import pprint

dataset_1 = {'database': './tests/test_data.csv', 'db_type': 'csv'}
datasets = [[dataset_1],]
org_ids = ids = [0,]

client = MockAlgorithmClient(
    datasets = datasets,
    organization_ids=org_ids,
    module='v6_basic_stats_py',
    collaboration_id=None,
    node_ids=None
)

organizations = client.organization.list()
org_ids = ids = [organization["id"] for organization in organizations]

average_task = client.task.create(
    input_={
        'method': 'flexible_stats',
        'kwargs': {
            'numeric_stats': ["count", "mean", "std", "median"],
            'categorical_stats': ["count", "unique"],
            'group_by': 'pat_id'
        }
    },
    organizations=[org_ids[0]]
)

results = client.result.get(average_task.get("id"))
df_events = pd.DataFrame(results)
print(df_events.head())
pprint(results)