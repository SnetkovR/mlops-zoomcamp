from datetime import datetime, timedelta
import os

import pickle
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import prefect
from prefect import flow, task
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import IntervalSchedule


FILE_MASK = 'fhv_tripdata_{year}-{month:02d}.parquet'
DATE_FORMAT = '%Y-%m-%d'
DEFAULT_PATH = 'data'


@task
def get_paths(date: str):
    converted_date = datetime.strptime(date, DATE_FORMAT)
    prev_date = converted_date - timedelta(days=converted_date.day)
    train_path = FILE_MASK.format(
        year=converted_date.year, month=converted_date.month)
    val_path = FILE_MASK.format(year=prev_date.year, month=prev_date.month)
    return os.path.join(DEFAULT_PATH, train_path), os.path.join(DEFAULT_PATH, val_path)


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df


@task
def prepare_features(df, categorical, train=True):
    logger = prefect.get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df, categorical):
    logger = prefect.get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    logger = prefect.get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return


@flow
def main(date="2021-08-15"):
    categorical = ['PUlocationID', 'DOlocationID']
    # train_path: str = './data/fhv_tripdata_2021-01.parquet', val_path: str = './data/fhv_tripdata_2021-02.parquet'
    train_path, val_path = get_paths(date).result()
    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    pickle.dump(lr, open(f'model-{date}.bin', 'wb'))
    pickle.dump(dv, open(f'dv-{date}.bin', 'wb'))
    run_model(df_val_processed, categorical, dv, lr)


DeploymentSpec(
    flow=main,
    name='model_training',
    schedule=IntervalSchedule(interval=timedelta(minutes=5)),
    flow_runner=SubprocessFlowRunner(),
    tags=['ml']
)
