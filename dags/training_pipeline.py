from __future__ import annotations

import pendulum

from airflow import DAG, Dataset
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.generic_transfer import GenericTransfer

from airflow.providers.sqlite.operators.sqlite import SQLExecuteQueryOperator


def generate_datasets():
    pass


with DAG(
        dag_id="suggested_order_training_pipeline",
        catchup=False,
        start_date=pendulum.datetime(2022, 12, 3, tz="UTC"),
        schedule="@weekly",
        tags=["training", "document-classification"],
) as dag1:
    # query the database to pull in training set
    # in reality the below would be a Redhift to S3 operator to unload the result set
    get_training_data = SQLExecuteQueryOperator(
        task_id="ingest_training_data",
        sql="select current_date",
    )

    # run the training script
    train_models = PythonOperator(
        task_id="train_models",
        python_callable=lambda x: "hello world",
    )

    # register the trained models
    register_models = SQLExecuteQueryOperator(
        task_id="register_models",
        sql="select current-date",
    )

    # get the evaluation dataset. This would be a Redshift to S3 operator to unload the result set
    evaluation_data = SQLExecuteQueryOperator(
        task_id="evaluation_dataset",
        sql="select current_date",
    )

    # get production models
    get_production_models = SQLExecuteQueryOperator(
        task_id="get_production_models",
        sql="select current_date",
    )

    # compare production model against the newly trained one
    evaluate_models = PythonOperator(
        task_id="evaluate_models",
        python_callable=lambda x: "hello world",
    )


    # update the production_model table to reflect which model_id has the best performance for the location
    update_production_model = SQLExecuteQueryOperator(
        task_id="update_production_model",
        sql="select current_date",
    )


    get_training_data >> train_models >> register_models >> evaluate_models >> update_production_model,
    evaluation_data >> evaluate_models >> update_production_model,
    get_production_models >> evaluate_models
