from prefect import flow, task

from prepare_data import download_data, find_and_delete_duplicates, split_data
from train import train as train_model


@task
def download_data_task():
    return download_data()


@task
def find_and_delete_duplicates_task(path):
    find_and_delete_duplicates(path)


@task
def split_data_task(path):
    split_data(path)


@task
def train_model_task():
    train_model()


@flow
def training_flow(name="crack-detection-training"):
    path = download_data_task()
    find_and_delete_duplicates_task(path)
    split_data_task(path)
    train_model_task()


if __name__ == "__main__":
    training_flow()
