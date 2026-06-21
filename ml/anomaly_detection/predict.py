"""
ml/anomaly_detection/predict.py
--------------------------------
Batch Isolation Forest inference using PySpark.
Loads a serialized Isolation Forest model from S3 and applies it
to each meter reading via a Spark UDF — scalable to millions of rows.

Anomaly scores < ANOMALY_THRESHOLD are flagged as anomalies.
Results are written to S3 and loaded into Redshift anomalies table.
"""

from __future__ import annotations

import logging
import os
import pickle
from datetime import date
from typing import Optional

import boto3
import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from ml.anomaly_detection.train import (
    ANOMALY_THRESHOLD,
    FEATURE_COLS,
    MODEL_S3_KEY,
    S3_BUCKET,
    engineer_features,
)

logger = logging.getLogger("anomaly_predict")

ANOMALY_OUTPUT_PREFIX = f"s3://{S3_BUCKET}/ml/anomaly_scores"
PROCESSED_PREFIX = f"s3://{S3_BUCKET}/processed/readings"


# ---------------------------------------------------------------------------
# Model loading (broadcast to Spark executors)
# ---------------------------------------------------------------------------
def load_model_from_s3() -> object:
    """Download and deserialize the Isolation Forest pipeline from S3."""
    s3 = boto3.client("s3")
    tmp = "/tmp/isolation_forest_model.pkl"
    s3.download_file(S3_BUCKET, MODEL_S3_KEY, tmp)
    with open(tmp, "rb") as f:
        pipeline = pickle.load(f)
    logger.info("Model loaded from s3://%s/%s", S3_BUCKET, MODEL_S3_KEY)
    return pipeline


# ---------------------------------------------------------------------------
# Spark UDF for scoring
# ---------------------------------------------------------------------------
def build_scoring_udf(pipeline):
    """
    Returns a Spark UDF that takes a struct of feature values
    and returns (anomaly_score: double, is_anomaly: boolean).
    """
    # Broadcast avoids serializing the model per row
    model = pipeline  # will be captured in closure

    def score_reading(*feature_values) -> float:
        """Compute anomaly score for a single reading."""
        x = np.array([[v if v is not None else 0.0 for v in feature_values]])
        score = model.named_steps["iso_forest"].score_samples(
            model.named_steps["scaler"].transform(x)
        )[0]
        return float(score)

    return F.udf(score_reading, T.DoubleType())


# ---------------------------------------------------------------------------
# Feature engineering via Pandas UDF for efficiency
# ---------------------------------------------------------------------------
@F.pandas_udf(T.StructType([
    T.StructField("reading_hour_sin", T.DoubleType()),
    T.StructField("reading_hour_cos", T.DoubleType()),
    T.StructField("day_of_week_sin", T.DoubleType()),
    T.StructField("day_of_week_cos", T.DoubleType()),
    T.StructField("apparent_power_kva", T.DoubleType()),
]))
def compute_cyclic_features(timestamp_series, pf_series, kwh_series):
    """Vectorized cyclical feature engineering."""
    import pandas as pd
    ts = pd.to_datetime(timestamp_series)
    hour = ts.dt.hour.astype(float)
    dow = ts.dt.dayofweek.astype(float)

    apparent = kwh_series / pf_series.replace(0, float("nan"))

    return pd.DataFrame({
        "reading_hour_sin": np.sin(2 * np.pi * hour / 24),
        "reading_hour_cos": np.cos(2 * np.pi * hour / 24),
        "day_of_week_sin": np.sin(2 * np.pi * dow / 7),
        "day_of_week_cos": np.cos(2 * np.pi * dow / 7),
        "apparent_power_kva": apparent,
    })


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------
def run_inference(
    spark: SparkSession,
    pipeline,
    process_date: date,
) -> DataFrame:
    """
    Load processed readings for the date, compute features,
    score each reading with Isolation Forest, and return a DataFrame
    with anomaly scores and flags.
    """
    date_str = process_date.isoformat()
    s3_path = f"{PROCESSED_PREFIX}/year={process_date.year}/month={process_date.month:02d}/day={process_date.day:02d}"
    logger.info("Loading readings from %s", s3_path)

    raw = spark.read.parquet(s3_path)
    logger.info("Loaded %d readings", raw.count())

    # Compute cyclical features via Pandas UDF
    cyclic = raw.select(
        "*",
        compute_cyclic_features(
            F.col("reading_timestamp"),
            F.col("power_factor"),
            F.col("kwh_consumed"),
        ).alias("cyclic"),
    ).select(
        "*",
        F.col("cyclic.reading_hour_sin"),
        F.col("cyclic.reading_hour_cos"),
        F.col("cyclic.day_of_week_sin"),
        F.col("cyclic.day_of_week_cos"),
        F.col("cyclic.apparent_power_kva"),
    )

    # Rolling window stats (7-day via Window function)
    w = Window.partitionBy("meter_id").orderBy("reading_timestamp").rowsBetween(-7 * 24, 0)
    cyclic = (
        cyclic
        .withColumn("rolling_7d_mean_kwh", F.avg("kwh_consumed").over(w))
        .withColumn("rolling_7d_std_kwh", F.stddev("kwh_consumed").over(w))
        .withColumn("kwh_z_score",
                    (F.col("kwh_consumed") - F.col("rolling_7d_mean_kwh"))
                    / F.greatest(F.col("rolling_7d_std_kwh"), F.lit(1e-9)))
    )

    # Build Spark UDF
    score_udf = build_scoring_udf(pipeline)

    # Score each reading
    feature_cols_expr = [F.col(c) for c in FEATURE_COLS]
    scored = (
        cyclic
        .withColumn("anomaly_score", score_udf(*feature_cols_expr))
        .withColumn("is_anomaly", F.col("anomaly_score") < F.lit(ANOMALY_THRESHOLD))
        .withColumn("anomaly_severity",
                    F.when(F.col("anomaly_score") < -0.30, "HIGH")
                    .when(F.col("anomaly_score") < -0.20, "MEDIUM")
                    .when(F.col("anomaly_score") < ANOMALY_THRESHOLD, "LOW")
                    .otherwise("NORMAL"))
        .withColumn("scored_at", F.current_timestamp())
    )

    anomaly_count = scored.filter(F.col("is_anomaly")).count()
    total = scored.count()
    anomaly_rate = round(anomaly_count / max(total, 1) * 100, 2)
    logger.info("Scored %d readings | Anomalies: %d (%.2f%%)", total, anomaly_count, anomaly_rate)

    return scored


# ---------------------------------------------------------------------------
# Write results
# ---------------------------------------------------------------------------
def write_anomaly_scores(df: DataFrame, process_date: date) -> str:
    """Write full scored DataFrame to S3 (for Redshift COPY)."""
    output_path = f"{ANOMALY_OUTPUT_PREFIX}/scoring_date={process_date.isoformat()}"
    (
        df
        .select(
            "meter_id", "reading_timestamp", "kwh_consumed", "voltage",
            "power_factor", "utility_zone", "customer_type",
            "anomaly_score", "is_anomaly", "anomaly_severity", "scored_at",
        )
        .repartition(10)
        .write
        .mode("overwrite")
        .option("compression", "snappy")
        .parquet(output_path)
    )
    logger.info("Anomaly scores written → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_batch_inference(spark: SparkSession, process_date: Optional[date] = None) -> dict:
    process_date = process_date or date.today()

    pipeline = load_model_from_s3()
    scored_df = run_inference(spark, pipeline, process_date)
    output_path = write_anomaly_scores(scored_df, process_date)

    anomaly_df = scored_df.filter(F.col("is_anomaly"))
    return {
        "process_date": process_date.isoformat(),
        "total_scored": scored_df.count(),
        "total_anomalies": anomaly_df.count(),
        "anomaly_rate_pct": round(anomaly_df.count() / max(scored_df.count(), 1) * 100, 2),
        "output_path": output_path,
        "status": "SUCCESS",
    }
