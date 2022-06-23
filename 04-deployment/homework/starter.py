import sys
import pickle
import pandas as pd

CATEGORICAL = ['PUlocationID', 'DOlocationID']


def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[CATEGORICAL] = df[CATEGORICAL].fillna(-1).astype('int').astype('str')

    return df


def run():
    if len(sys.argv) < 2:
        raise AttributeError('Not enough arguments!')
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    df = read_data(
        f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    )
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[CATEGORICAL].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print(f"Mean for year {year} month {month} {y_pred.mean()}")
    df['prediction'] = y_pred
    df_result = df[['ride_id', 'prediction']]
    df_result.to_parquet(
        f"predictions_{year:04d}-{month:02d}.parquet",
        engine='pyarrow',
        compression=None,
        index=False
    )


if __name__ == '__main__':
    run()
