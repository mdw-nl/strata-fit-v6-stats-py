from functools import wraps
from pydantic import BaseModel, ValidationError, Field
from typing import Dict, Union, Optional, Any

def enforce_output_schema(model: BaseModel):
    """
    Decorator to validate output against a Pydantic model.
    Sanitizes validation errors to avoid leaking sensitive node data.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            try:
                validated = model.model_validate(result)
            except ValidationError as e:

                # Extract field name only (first in the error array) and no value
                error_locations = set(
                    f"Error type: {err['type']}, error location: {err.get('loc', [])[0]}" if err.get('loc') else 'unknown_field'
                    for err in e.errors()
                )


                safe_error_message = (
                    f"Validation error in function '{func.__name__}' "
                    f"while validating model '{model.__name__}'. "
                    f"Issue detected in fields: \n\t{'\n\t'.join(error_locations) or 'unknown fields'}."
                )

                raise ValueError(safe_error_message) from None
            
            # Handle unexpected exceptions here
            except Exception as e:
                safe_error_message = (
                    f"Validation of model '{model.__name__}' failed unexpectedly "
                    f"during the run of function '{func.__name__}'. "
                    f"Error type is '{type(e)}', error message is hidden for security reasons."
                )
                raise Exception(safe_error_message) from None

            # Handle potential serialization errors
            try:
                return validated.model_dump()
            except Exception as e:
                safe_error_message = (
                    f"Data dump for model '{model.__name__}' failed unexpectedly "
                    f"during the run of function '{func.__name__}'. "
                    f"Error type is '{type(e)}', error message is hidden for security reasons."
                )
                raise Exception(safe_error_message) from None


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
    RF_positivity_counts: Dict[Union[int, str, float], Union[int, str]]
    RF_positivity_proportions: Dict[Union[int, str, float], Union[float, str]]
    anti_CCP_counts: Dict[Union[int, str, float], Union[int, str]]
    anti_CCP_proportions: Dict[Union[int, str, float], Union[float, str]]

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
    Q1: float
    Q3: float

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