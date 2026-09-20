from prefect import flow, get_run_logger, task

from calculate_drift import check_drift, get_current_data, get_reference_data
from training_flow import training_flow


@task(name="Fetch Production & Reference Data")
def get_monitoring_datasets(bucket_name: str):
    logger = get_run_logger()
    logger.info("Downloading reference and current datasets...")
    ref_df = get_reference_data()
    cur_df = get_current_data(bucket_name)
    return ref_df, cur_df


@task(name="Calculate Data Drift")
def run_drift_check(reference_df, current_df):
    logger = get_run_logger()
    logger.info("Calculating data drift via Evidently...")
    drift_result = check_drift(reference_df, current_df)
    return drift_result


@flow(name="Concrete Crack Monitoring Flow")
def monitoring_flow(bucket_name: str = "crack-detection-mlflow-s3-bucket", threshold: float = 0.5):
    logger = get_run_logger()

    reference_dataset, current_dataset = get_monitoring_datasets(bucket_name)

    if current_dataset.empty:
        logger.warning("Current dataset is empty. Skipping drift check.")
        return

    drift_result = run_drift_check(reference_dataset, current_dataset)

    logger.info(f"Drift share: {drift_result['drift_share']:.2%}, Threshold: {threshold:.2%}")

    if drift_result["drift_share"] >= threshold:
        logger.warning("Data drift detected above threshold. Triggering retraining flow...")
        training_flow()
    else:
        logger.info("No significant drift detected. Retraining not needed.")


if __name__ == "__main__":
    monitoring_flow()
