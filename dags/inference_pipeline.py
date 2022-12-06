from __future__ import annotations

import pendulum

from airflow import DAG, Dataset
from airflow.decorators.python import TaskDecorator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.generic_transfer import GenericTransfer

from airflow.providers.sqlite.operators.sqlite import SQLExecuteQueryOperator


def generate_datasets():
    pass


with DAG(
        dag_id="suggested_order_forecasting_pipeline",
        catchup=False,
        start_date=pendulum.datetime(2022, 12, 3, tz="UTC"),
        schedule="@weekly",
        tags=["training", "suggested-orders"],
) as dag1:
    # where the inference engine looks for model artifacts
    get_model_metadata = SQLExecuteQueryOperator(
        task_id="query_production_models",
        sql="select current_date"
    )

    # ingest the data input for forecasting... in reality this query does select and unload of the result set into s3
    ingest_inference_data = SQLExecuteQueryOperator(
        task_id="query_inference_input",
        sql="select current_date"
    )

    # run the inference script
    inference = PythonOperator(
        task_id="inference",
        python_callable= lambda _: "hello world"
    )

    # push the inference into Redshift
    publish_inference = SQLExecuteQueryOperator(
        task_id="push_inference_to_redshift",
        sql="select current_date"
    )

    (get_model_metadata >> inference), (ingest_inference_data >> inference),  inference >> publish_inference

