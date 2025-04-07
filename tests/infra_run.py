import json
from vantage6.client import Client

from getpass import getpass
from pydantic import BaseModel, validator
from typing import Optional
from dynaconf import Dynaconf

class Vantage6Config(BaseModel):
    server_url: str
    server_port: int
    server_api: str
    username: str
    password: str
    ci_mode: bool = False
    organization_key: Optional[str] = None  # Optional if you want to use encryption

    @validator('server_url')
    def url_must_be_valid(cls, v):
        if not v.startswith("http"):
            raise ValueError("server_url must start with http")
        return v

# Load settings via Dynaconf from settings files or environment variables.
settings = Dynaconf(
    settings_files=['settings/infra.toml', 'settings/.secrets.toml'],
    envvar_prefix="V6"
)

# Prepare configuration data. In CI mode, the password should come from settings;
# otherwise, we prompt for it.
config = Vantage6Config(
    server_url = settings.get("SERVER_URL", "http://localhost"),
    server_port = settings.get("SERVER_PORT", 5070),
    server_api = settings.get("SERVER_API", "/api"),
    username = settings.get("USERNAME", "gamma-user"),
    ci_mode = settings.get("CI_MODE", False),
    password = settings.get("PASSWORD") if settings.get("CI_MODE", False) else getpass("Enter <PASSWORD>: "),
    organization_key = settings.get("ORGANIZATION_KEY")
)


def run_task(algorithm_config):
    # Initialize client and authenticate using our config object
    client = Client(config.server_url, config.server_port, config.server_api)
    client.authenticate(username=config.username, password=config.password)
    if "organization_key" in config:
        client.setup_encryption(config.organization_key)

    # Retrieve organizations and perform an intermediary check
    organizations_data = client.organization.list().get('data', [])
    organizations = {org['name']: org['id'] for org in organizations_data}
    if not organizations:
        raise RuntimeError("No organizations found!")
    
    # Retrieve collaborations and check if they exist
    collaborations = client.collaboration.list().get('data', [])
    if not collaborations:
        raise RuntimeError("No collaborations found!")
    
    print(algorithm_config)

    # Create and run the task
    task = client.task.create(
        image=algorithm_config['image'],
        name=algorithm_config['name'],
        description=algorithm_config['description'],
        input_=algorithm_config['input'],
        organizations=algorithm_config['organizations'],
        collaboration=algorithm_config['collaboration'],
        databases=algorithm_config['databases']
    )
    
    task_id = task.get("id")
    print(task_id)
    client.wait_for_results(task_id)
    results = client.result.get(task_id)
    return results

if __name__ == "__main__":
    # Define algorithm-specific configurations as dictionaries.
    # km_config = {
    #     'image': 'harbor2.vantage6.ai/algorithms/kaplan-meier',
    #     'name': 'demo-km-analysis',
    #     'description': 'Kaplan-Meier dry-run',
    #     'input': {
    #         'method': 'kaplan_meier_central',
    #         'kwargs': {
    #             'time_column_name': 'Survival.time',
    #             'censor_column_name': 'deadstatus.event',
    #             'organizations_to_include': [1, 2, 3]
    #         }
    #     },
    #     'organizations': [2],
    #     'collaboration': 1,
    #     'databases': [{'label': 'default'}]
    # }

    # avg_config = {
    #     'image': 'ghcr.io/mdw-nl/v6-average-py:v1.0.1',
    #     'name': 'demo-average',
    #     'description': 'Average dry-run',
    #     'input': {
    #         'method': 'central_average',
    #         'kwargs': {
    #             'column_name': ['age'],
    #             'org_ids': [1, 2, 3]
    #         }
    #     },
    #     'organizations': [2],
    #     'collaboration': 1,
    #     'databases': [{'label': 'default'}]
    # }
    
    stats_config = {
        'image': 'ghcr.io/mdw-nl/strata-fit-v6-stats-py@sha256:ba2457cdf6f1917fcbfcf65f9297a4664232227f6005334b29a20805afad4eb3',
        'name': 'demo-stats',
        'description': 'Stats dry-run',
        'input': {
            'method': 'partial_stats',
            'kwargs': {}
        },
        'organizations': [2],
        'collaboration': 1,
        'databases': [{'label': 'default'}]
    }

    # Run the tasks. You can choose to run one or both algorithms.
    results_stats = run_task(stats_config)

    # Instead of printing results directly, we collect them in a JSON dictionary.
    output = {
        "stats_results": results_stats,
    }

    #print(json.dumps(output, indent=2))
