from functools import wraps
from pydantic import BaseModel, ValidationError, Field
from typing import Dict, Union, Optional, Any

def enforce_output_schema(model: BaseModel):
    """
    A decorator that validates the output of a function against a Pydantic model.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            try:
                validated = model.model_validate(result)
            except ValidationError as e:
                raise ValueError(f"Output validation error in {func.__name__}: {e}")
            return validated.model_dump()
        return wrapper
    return decorator

# ------------------------------
# Pydantic Models for Outputs
# ------------------------------

class UniquePatientsOutput(BaseModel):
    # Our unique_patients function returns either an int or a masked string (e.g., "<5")
    unique_patients: Union[int, str]

class VisitDefinitionOutput(BaseModel):
    invalid_visits: int

class VisitsPerTimePeriodOutput(BaseModel):
    visit_rate_mean: Optional[float]
    visit_rate_std: Optional[float]
    visit_rate_median: Optional[float]
    total_patients: int

class MissingDataPerVisitOutput(BaseModel):
    count_all_missing: int
    total_visits: int
    percent_all_missing: float

class DemographicsOutput(BaseModel):
    Age_mean: float
    Age_std: float
    Sex_counts: Dict[Union[int, str], Union[int, str]]
    Sex_proportions: Dict[Union[int, str], Union[float, str]]
    RF_positivity_counts: Dict[Union[int, str], Union[int, str]]
    RF_positivity_proportions: Dict[Union[int, str], Union[float, str]]
    anti_CCP_counts: Dict[Union[int, str], Union[int, str]]
    anti_CCP_proportions: Dict[Union[int, str], Union[float, str]]

class DiseaseDurationDistributionOutput(BaseModel):
    Year_diagnosis_mean: float
    Year_diagnosis_std: float
    Year_diagnosis_skewness: float

# For Laboratory Values Overall, each lab variable's output
class LabValueOverall(BaseModel):
    mean: float
    std: float
    skewness: float
    outlier_count: int

# For Laboratory Values Aggregated, we need to capture the 25th and 75th percentiles.
class LabValueAggregated(BaseModel):
    mean: float
    std: float
    median: float
    Q1: float = Field(..., alias="25%")
    Q3: float = Field(..., alias="75%")

class LabValueOverallOutput(BaseModel):
    CRP: LabValueOverall
    ESR: LabValueOverall
    TJC28: LabValueOverall
    SJC28: LabValueOverall
    DAS28: LabValueOverall
    Pat_global: LabValueOverall
    Ph_global: LabValueOverall
    Pain: LabValueOverall

class LabValueAggregatedOutput(BaseModel):
    CRP: LabValueAggregated
    ESR: LabValueAggregated
    TJC28: LabValueAggregated
    SJC28: LabValueAggregated
    DAS28: LabValueAggregated
    Pat_global: LabValueAggregated
    Ph_global: LabValueAggregated
    Pain: LabValueAggregated


# Finally, a composite model for the overall algorithm output
class PartialStatsOutput(BaseModel):
    unique_patients_per_center: UniquePatientsOutput
    check_visit_definition: VisitDefinitionOutput
    visits_per_time_period: VisitsPerTimePeriodOutput
    missing_data_per_visit: MissingDataPerVisitOutput
    demographics: DemographicsOutput
    disease_duration_distribution: DiseaseDurationDistributionOutput
    laboratory_values_overall: LabValueOverallOutput
    laboratory_values_grouped_by_pat_id: LabValueAggregatedOutput