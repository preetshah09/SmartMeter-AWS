"""
dags/smartmeter_pipeline_dag.py
--------------------------------
Airflow DAG orchestrating the full SmartMeter pipeline:
  1. Trigger EMR Spark job (data processing)
  2. Run anomaly detection batch inference
  3. Load results into Redshift
  4. Run data quality validation
  5. Notify on anomaly spike

Runs daily at 02:00 UTC.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.emr import EmrAddStepsOperator
from airflow.providers.amazon.aws.sensors.emr import EmrStepSensor
from airflow.utils.trigger_rule import TriggerRule

DEFAULT_ARGS = {
    "owner": "preet_shah",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": True,
    "email": ["shahpreetp15@gmail.com"],
}

# ---------------------------------------------------------------------------
# AWS config (from Airflow Variables)
# ---------------------------------------------------------------------------
EMR_CLUSTER_ID = Variable.get("EMR_CLUSTER_ID", default_var="j-PLACEHOLDER")
S3_BUCKET = Variable.get("S3_BUCKET", default_var="smartmeter-data-lake")
REDSHIFT_CONN_ID = "redshift_smartmeter"
AWS_CONN_ID = "aws_default"

# ---------------------------------------------------------------------------
# EMR Steps
# ---------------------------------------------------------------------------
SPARK_PROCESSING_STEP = [
    {
        "Name": "SmartMeter-ProcessReadings",
        "ActionOnFailure": "TERMINATE_CLUSTER",
        "HadoopJarStep": {
            "Jar": "command-runner.jar",
            "Args": [
                "spark-submit",
                "--deploy-mode", "cluster",
                "--master", "yarn",
                "--conf", "spark.sql.adaptive.enabled=true",
                "--py-files", f"s3://{S3_BUCKET}/code/processing.zip",
                f"s3://{S3_BUCKET}/code/emr_spark_job.py",
                "--date", "{{ ds }}",
            ],
        },
    }
]

ANOMALY_DETECTION_STEP = [
    {
        "Name": "SmartMeter-AnomalyDetection",
        "ActionOnFailure": "CONTINUE",
        "HadoopJarStep": {
            "Jar": "command-runner.jar",
            "Args": [
                "spark-submit",
                "--deploy-mode", "cluster",
                "--master", "yarn",
                "--py-files", f"s3://{S3_BUCKET}/code/processing.zip",
                f"s3://{S3_BUCKET}/code/run_inference.py",
                "--date", "{{ ds }}",
            ],
        },
    }
]


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------
def _load_hourly_to_redshift(**context) -> None:
    """COPY hourly aggregation from S3 to Redshift."""
    import psycopg2
    ds = context["ds"]  # YYYY-MM-DD
    s3_path = f"s3://{S3_BUCKET}/processed/hourly/reading_date={ds}/"
    iam_role = Variable.get("REDSHIFT_IAM_ROLE")

    conn = psycopg2.connect(
        host=Variable.get("REDSHIFT_HOST"),
        port=5439,
        dbname=Variable.get("REDSHIFT_DB", default_var="smartmeter"),
        user=Variable.get("REDSHIFT_USER"),
        password=Variable.get("REDSHIFT_PASSWORD"),
    )
    with conn.cursor() as cur:
        cur.execute(f"""
            COPY smartmeter.hourly_readings
            FROM '{s3_path}'
            IAM_ROLE '{iam_role}'
            FORMAT AS PARQUET
            FILLRECORD;
        """)
        conn.commit()
    conn.close()


def _load_anomalies_to_redshift(**context) -> None:
    """COPY anomaly scores from S3 to Redshift."""
    import psycopg2
    ds = context["ds"]
    s3_path = f"s3://{S3_BUCKET}/ml/anomaly_scores/scoring_date={ds}/"
    iam_role = Variable.get("REDSHIFT_IAM_ROLE")

    conn = psycopg2.connect(
        host=Variable.get("REDSHIFT_HOST"),
        port=5439,
        dbname=Variable.get("REDSHIFT_DB", default_var="smartmeter"),
        user=Variable.get("REDSHIFT_USER"),
        password=Variable.get("REDSHIFT_PASSWORD"),
    )
    with conn.cursor() as cur:
        cur.execute(f"""
            COPY smartmeter.anomaly_scores
            FROM '{s3_path}'
            IAM_ROLE '{iam_role}'
            FORMAT AS PARQUET
            FILLRECORD;
        """)
        conn.commit()
    conn.close()


def _run_dq_checks(**context) -> None:
    """Validate Redshift tables post-load."""
    import psycopg2
    ds = context["ds"]

    conn = psycopg2.connect(
        host=Variable.get("REDSHIFT_HOST"),
        port=5439,
        dbname=Variable.get("REDSHIFT_DB", default_var="smartmeter"),
        user=Variable.get("REDSHIFT_USER"),
        password=Variable.get("REDSHIFT_PASSWORD"),
    )
    checks = [
        # 1. Hourly readings loaded for the date
        f"SELECT COUNT(*) FROM smartmeter.hourly_readings WHERE reading_date = '{ds}'",
        # 2. No duplicate meter+date+hour combos
        f"""SELECT COUNT(*) FROM (
                SELECT meter_id, reading_date, reading_hour, COUNT(*)
                FROM smartmeter.hourly_readings
                WHERE reading_date = '{ds}'
                GROUP BY 1,2,3 HAVING COUNT(*) > 1
            ) dupes""",
    ]
    with conn.cursor() as cur:
        for sql in checks:
            cur.execute(sql)
            result = cur.fetchone()[0]
            if "dupes" in sql and result > 0:
                raise ValueError(f"DQ FAILED: {result} duplicate meter-hour combos on {ds}")
            elif "COUNT(*) FROM smartmeter" in sql and result == 0:
                raise ValueError(f"DQ FAILED: 0 rows loaded for {ds}")
    conn.close()
    print(f"✅ DQ checks passed for {ds}")


def _check_anomaly_rate(**context) -> None:
    """Alert if anomaly rate spikes above 5% (possible grid event)."""
    import psycopg2
    ds = context["ds"]

    conn = psycopg2.connect(
        host=Variable.get("REDSHIFT_HOST"),
        port=5439,
        dbname=Variable.get("REDSHIFT_DB", default_var="smartmeter"),
        user=Variable.get("REDSHIFT_USER"),
        password=Variable.get("REDSHIFT_PASSWORD"),
    )
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) AS anomalies,
                ROUND(
                    100.0 * SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2
                ) AS anomaly_rate_pct
            FROM smartmeter.anomaly_scores
            WHERE DATE(scoring_date) = '{ds}'
        """)
        row = cur.fetchone()
    conn.close()

    if row:
        total, anomalies, rate = row
        print(f"Anomaly rate for {ds}: {rate}% ({anomalies}/{total})")
        if rate and rate > 5.0:
            raise ValueError(f"ALERT: Anomaly rate {rate}% exceeds 5% threshold on {ds} – possible grid event!")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="smartmeter_pipeline",
    description="Daily SmartMeter processing + anomaly detection pipeline on AWS",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 2 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["smartmeter", "aws", "emr", "redshift", "ml"],
) as dag:

    start = EmptyOperator(task_id="start")

    # Step 1: EMR Spark processing
    submit_spark_job = EmrAddStepsOperator(
        task_id="submit_spark_processing",
        job_flow_id=EMR_CLUSTER_ID,
        steps=SPARK_PROCESSING_STEP,
        aws_conn_id=AWS_CONN_ID,
    )

    wait_for_spark = EmrStepSensor(
        task_id="wait_for_spark_processing",
        job_flow_id=EMR_CLUSTER_ID,
        step_id="{{ task_instance.xcom_pull('submit_spark_processing', key='return_value')[0] }}",
        aws_conn_id=AWS_CONN_ID,
        poke_interval=60,
        timeout=3600,
    )

    # Step 2: Anomaly detection
    submit_anomaly_job = EmrAddStepsOperator(
        task_id="submit_anomaly_detection",
        job_flow_id=EMR_CLUSTER_ID,
        steps=ANOMALY_DETECTION_STEP,
        aws_conn_id=AWS_CONN_ID,
    )

    wait_for_anomaly = EmrStepSensor(
        task_id="wait_for_anomaly_detection",
        job_flow_id=EMR_CLUSTER_ID,
        step_id="{{ task_instance.xcom_pull('submit_anomaly_detection', key='return_value')[0] }}",
        aws_conn_id=AWS_CONN_ID,
        poke_interval=60,
        timeout=1800,
    )

    # Step 3: Load to Redshift
    load_hourly = PythonOperator(task_id="load_hourly_to_redshift", python_callable=_load_hourly_to_redshift)
    load_anomalies = PythonOperator(task_id="load_anomalies_to_redshift", python_callable=_load_anomalies_to_redshift)

    # Step 4: DQ checks
    dq_checks = PythonOperator(task_id="run_dq_checks", python_callable=_run_dq_checks)
    anomaly_rate_check = PythonOperator(task_id="check_anomaly_rate", python_callable=_check_anomaly_rate)

    end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

    # Flow
    (
        start
        >> submit_spark_job
        >> wait_for_spark
        >> submit_anomaly_job
        >> wait_for_anomaly
        >> [load_hourly, load_anomalies]
        >> dq_checks
        >> anomaly_rate_check
        >> end
    )
