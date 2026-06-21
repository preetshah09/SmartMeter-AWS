"""
processing/emr_spark_job.py
----------------------------
PySpark job running on AWS EMR.
Processes 3M+ hourly smart meter records from S3:
  - Reads raw Parquet from S3
  - Applies partition pruning + clustering
  - Aggregates to hourly/daily summaries
  - Writes optimized Parquet back to S3 for Redshift COPY

Usage (EMR step):
    spark-submit \
        --deploy-mode cluster \
        --py-files s3://smartmeter-data-lake/code/processing.zip \
        s3://smartmeter-data-lake/code/emr_spark_job.py \
        --date 2024-01-15
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import date, datetime, timedelta
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")
logger = logging.getLogger("smartmeter_emr")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
S3_BUCKET = os.environ.get("S3_BUCKET", "smartmeter-data-lake")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

RAW_S3_PREFIX = f"s3://{S3_BUCKET}/raw/meter_readings"
PROCESSED_S3_PREFIX = f"s3://{S3_BUCKET}/processed"
ANOMALY_S3_PREFIX = f"s3://{S3_BUCKET}/ml/anomaly_scores"


# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------
def get_spark(app_name: str = "SmartMeter-EMR") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.parquet.filterPushdown", "true")       # Partition pruning
        .config("spark.sql.files.maxPartitionBytes", "134217728")  # 128 MB
        .config("spark.sql.shuffle.partitions", "400")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# Raw schema
# ---------------------------------------------------------------------------
RAW_SCHEMA = T.StructType([
    T.StructField("meter_id", T.StringType(), False),
    T.StructField("reading_timestamp", T.TimestampType(), False),
    T.StructField("kwh_consumed", T.DoubleType(), True),
    T.StructField("voltage", T.DoubleType(), True),
    T.StructField("current_amps", T.DoubleType(), True),
    T.StructField("power_factor", T.DoubleType(), True),
    T.StructField("reactive_power_kvar", T.DoubleType(), True),
    T.StructField("temperature_celsius", T.DoubleType(), True),
    T.StructField("meter_status", T.StringType(), True),
    T.StructField("utility_zone", T.StringType(), True),
    T.StructField("customer_type", T.StringType(), True),  # RESIDENTIAL, COMMERCIAL, INDUSTRIAL
    T.StructField("lat", T.DoubleType(), True),
    T.StructField("lon", T.DoubleType(), True),
])


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def read_raw(spark: SparkSession, process_date: date, lookback_days: int = 0) -> DataFrame:
    """
    Read raw Parquet with partition pruning.
    Pushes date filters down to S3 partition paths to avoid full scans.
    """
    paths = []
    for i in range(lookback_days + 1):
        d = process_date - timedelta(days=i)
        paths.append(f"{RAW_S3_PREFIX}/year={d.year}/month={d.month:02d}/day={d.day:02d}/")

    logger.info("Reading %d partition(s): %s … %s", len(paths), paths[-1], paths[0])

    df = (
        spark.read
        .schema(RAW_SCHEMA)
        .option("mergeSchema", "true")
        .parquet(*paths)
    )

    # Filter to exact date (partition pruning already narrowed the scan)
    df = df.filter(F.to_date(F.col("reading_timestamp")) == F.lit(process_date.isoformat()))

    logger.info("Raw rows for %s: %d", process_date, df.count())
    return df


# ---------------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------------
def clean_readings(df: DataFrame) -> DataFrame:
    """Apply data quality rules and feature engineering."""
    # Basic cleaning
    df = (
        df
        .filter(F.col("meter_id").isNotNull())
        .filter(F.col("reading_timestamp").isNotNull())
        .filter(F.col("kwh_consumed") >= 0)               # No negative consumption
        .filter(F.col("voltage").between(90, 280))         # Plausible voltage range
        .filter(F.col("power_factor").between(0.0, 1.0))  # PF must be 0–1
        .dropDuplicates(["meter_id", "reading_timestamp"])
    )

    # Feature engineering
    df = (
        df
        .withColumn("reading_hour", F.hour(F.col("reading_timestamp")))
        .withColumn("reading_date", F.to_date(F.col("reading_timestamp")))
        .withColumn("is_peak_hour", F.when(F.col("reading_hour").between(17, 21), True).otherwise(False))
        .withColumn("is_off_peak", F.when(F.col("reading_hour").between(0, 6), True).otherwise(False))
        .withColumn("apparent_power_kva",
                    F.when(F.col("power_factor") > 0,
                           F.col("kwh_consumed") / F.col("power_factor"))
                    .otherwise(F.lit(None).cast(T.DoubleType())))
        .withColumn("_processed_at", F.current_timestamp())
    )

    return df


def aggregate_hourly(df: DataFrame) -> DataFrame:
    """Hourly rollup per meter."""
    window_7d = Window.partitionBy("meter_id").orderBy("reading_timestamp").rowsBetween(-6, 0)

    hourly = (
        df
        .groupBy("meter_id", "reading_date", "reading_hour", "utility_zone", "customer_type")
        .agg(
            F.sum("kwh_consumed").alias("total_kwh"),
            F.avg("kwh_consumed").alias("avg_kwh"),
            F.max("kwh_consumed").alias("max_kwh"),
            F.min("kwh_consumed").alias("min_kwh"),
            F.stddev("kwh_consumed").alias("std_kwh"),
            F.avg("voltage").alias("avg_voltage"),
            F.min("voltage").alias("min_voltage"),
            F.max("voltage").alias("max_voltage"),
            F.avg("power_factor").alias("avg_power_factor"),
            F.avg("current_amps").alias("avg_current"),
            F.avg("temperature_celsius").alias("avg_temp_c"),
            F.count("*").alias("reading_count"),
            F.sum(F.col("is_peak_hour").cast(T.IntegerType())).alias("peak_hour_readings"),
        )
    )

    logger.info("Hourly aggregation rows: %d", hourly.count())
    return hourly


def aggregate_daily(df: DataFrame) -> DataFrame:
    """Daily rollup per meter — used for Redshift analytics layer."""
    daily = (
        df
        .groupBy("meter_id", "reading_date", "utility_zone", "customer_type")
        .agg(
            F.sum("kwh_consumed").alias("daily_kwh"),
            F.avg("kwh_consumed").alias("avg_hourly_kwh"),
            F.max("kwh_consumed").alias("peak_kwh"),
            F.avg("voltage").alias("avg_voltage"),
            F.avg("power_factor").alias("avg_power_factor"),
            F.count("*").alias("total_readings"),
            F.sum(F.col("is_peak_hour").cast(T.IntegerType())).alias("peak_readings"),
            F.sum(F.col("is_off_peak").cast(T.IntegerType())).alias("off_peak_readings"),
        )
    )

    logger.info("Daily aggregation rows: %d", daily.count())
    return daily


# ---------------------------------------------------------------------------
# S3 writer (optimized Parquet)
# ---------------------------------------------------------------------------
def write_parquet(df: DataFrame, s3_path: str, partition_cols: list[str], num_partitions: int = 10):
    """Write Parquet to S3 with clustering (repartition by partition_cols)."""
    logger.info("Writing → %s (partitions: %s)", s3_path, partition_cols)
    (
        df
        .repartition(num_partitions, *[F.col(c) for c in partition_cols])
        .write
        .mode("overwrite")
        .partitionBy(*partition_cols)
        .option("compression", "snappy")
        .parquet(s3_path)
    )
    logger.info("✅ Write complete: %s", s3_path)


# ---------------------------------------------------------------------------
# Data quality report
# ---------------------------------------------------------------------------
def run_data_quality_checks(raw_df: DataFrame, cleaned_df: DataFrame, process_date: date) -> dict:
    """Compare raw vs cleaned counts and flag anomalies."""
    raw_count = raw_df.count()
    clean_count = cleaned_df.count()
    drop_pct = round((1 - clean_count / max(raw_count, 1)) * 100, 2)

    null_checks = {
        col: cleaned_df.filter(F.col(col).isNull()).count()
        for col in ["kwh_consumed", "voltage", "meter_id"]
    }

    report = {
        "process_date": process_date.isoformat(),
        "raw_count": raw_count,
        "clean_count": clean_count,
        "dropped_pct": drop_pct,
        "null_checks": null_checks,
        "status": "WARN" if drop_pct > 5 else "OK",
    }
    logger.info("DQ Report: %s", report)

    if drop_pct > 10:
        raise RuntimeError(f"DQ check FAILED: {drop_pct}% rows dropped – exceeds 10% threshold")

    return report


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------
def run_pipeline(process_date: date, lookback_days: int = 0) -> dict:
    spark = get_spark()

    logger.info("Starting SmartMeter pipeline for %s", process_date)

    # Read
    raw_df = read_raw(spark, process_date, lookback_days)

    # Clean
    cleaned_df = clean_readings(raw_df)

    # DQ
    dq_report = run_data_quality_checks(raw_df, cleaned_df, process_date)

    # Aggregate
    hourly_df = aggregate_hourly(cleaned_df)
    daily_df = aggregate_daily(cleaned_df)

    # Write cleaned readings (input for ML)
    write_parquet(
        cleaned_df,
        f"{PROCESSED_S3_PREFIX}/readings/year={process_date.year}/month={process_date.month:02d}/day={process_date.day:02d}",
        partition_cols=["utility_zone", "customer_type"],
        num_partitions=20,
    )

    # Write hourly aggregation
    write_parquet(
        hourly_df,
        f"{PROCESSED_S3_PREFIX}/hourly/reading_date={process_date.isoformat()}",
        partition_cols=["utility_zone"],
        num_partitions=5,
    )

    # Write daily aggregation
    write_parquet(
        daily_df,
        f"{PROCESSED_S3_PREFIX}/daily/reading_date={process_date.isoformat()}",
        partition_cols=["customer_type"],
        num_partitions=3,
    )

    logger.info("Pipeline complete for %s", process_date)
    spark.stop()

    return {
        "process_date": process_date.isoformat(),
        "raw_rows": dq_report["raw_count"],
        "clean_rows": dq_report["clean_count"],
        "drop_pct": dq_report["dropped_pct"],
        "status": "SUCCESS",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartMeter EMR Spark Job")
    parser.add_argument("--date", required=False, default=date.today().isoformat(),
                        help="Processing date YYYY-MM-DD")
    parser.add_argument("--lookback-days", type=int, default=0,
                        help="Additional past days to include")
    args = parser.parse_args()
    run_pipeline(
        process_date=date.fromisoformat(args.date),
        lookback_days=args.lookback_days,
    )
