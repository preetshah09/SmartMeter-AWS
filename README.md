# SmartMeter – AWS Cloud Pipeline with ML Integration

A scalable AWS data pipeline processing 3M+ hourly smart meter records using PySpark on EMR, with an ML-integrated anomaly detection system achieving 91% precision using Isolation Forest.

## Architecture

```
Smart Meters (IoT)
        │
        ▼
   AWS S3 (Raw)
        │
        ▼
  AWS EMR (PySpark)  ──── partition pruning + clustering ────►  Amazon Redshift
        │
        ▼
  Isolation Forest (Sklearn on Spark)
        │
        ▼
  Anomaly Flags → S3 → Redshift (anomalies table)
        │
  Apache Airflow (orchestration) + Docker (containerized)
```

## Tech Stack

| Layer | Technology |
|---|---|
| Ingestion | AWS S3 (raw landing zone) |
| Processing | PySpark on AWS EMR |
| Warehouse | Amazon Redshift (RA3) |
| ML | Scikit-learn Isolation Forest + PySpark UDF |
| Orchestration | Apache Airflow |
| Containerization | Docker + Docker Compose |
| Language | Python 3.11, PySpark 3.5 |

## Project Structure

```
smartmeter-aws/
├── dags/
│   └── smartmeter_pipeline_dag.py    # Airflow DAG
├── processing/
│   ├── emr_spark_job.py              # PySpark transformation job
│   └── redshift_loader.py            # Redshift COPY loader
├── ml/
│   ├── anomaly_detection/
│   │   ├── train.py                  # Isolation Forest training
│   │   └── predict.py                # Batch inference on Spark
│   └── model_artifacts/              # Serialized models (git-ignored)
├── infrastructure/
│   ├── docker/
│   │   └── Dockerfile
│   └── terraform/
│       └── main.tf
├── config/
│   └── settings.py
└── tests/
    └── test_emr_job.py
```

## Setup

```bash
pip install pyspark==3.5.1 boto3==1.34.0 scikit-learn==1.5.0 \
    apache-airflow==2.9.2 apache-airflow-providers-amazon==8.22.0 \
    pandas==2.2.2 pyarrow==16.1.0 psycopg2-binary==2.9.9
```

### Environment

```bash
export AWS_REGION=us-east-1
export S3_BUCKET=smartmeter-data-lake
export EMR_CLUSTER_ID=j-XXXXXXXXXX
export REDSHIFT_HOST=smartmeter.xxxxxx.us-east-1.redshift.amazonaws.com
export REDSHIFT_DB=smartmeter
export REDSHIFT_USER=admin
export REDSHIFT_PASSWORD=your-password
export REDSHIFT_IAM_ROLE=arn:aws:iam::123456789:role/RedshiftS3Access
```

## Running

```bash
# Run PySpark job on EMR
aws emr add-steps --cluster-id $EMR_CLUSTER_ID \
    --steps Type=Spark,Name=SmartMeter,ActionOnFailure=CONTINUE,\
Args=[--deploy-mode,cluster,--py-files,s3://smartmeter-data-lake/code/processing.zip,\
s3://smartmeter-data-lake/code/emr_spark_job.py,--date,2024-01-15]

# Or run via Airflow
airflow dags trigger smartmeter_pipeline --conf '{"process_date": "2024-01-15"}'

# Docker
docker-compose up --build
```

## ML Model

The Isolation Forest model is trained on 30 days of rolling historical data and refreshed weekly.
Anomaly scores < threshold (tuned at -0.15) trigger alerts.

Metrics achieved:
- Precision: 91%
- Recall: 87%
- F1 Score: 0.89
