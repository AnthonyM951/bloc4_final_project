"""Batch analytics pipeline for computing KPIs with Apache Spark.

This module is intended to be executed with ``spark-submit``. It connects to
our PostgreSQL/Supabase database, loads the ``jobs`` table in a distributed
manner and computes aggregated metrics that can be consumed by the admin
analytics dashboard.

Typical usage::

    SPARK_KPI_OUTPUT_PATH=s3a://analytics/kpi_daily \
    spark-submit src/pipelines/spark_kpis.py

The script is configurable through environment variables so it can run on a
local machine (using the embedded Spark master) or a remote cluster managed by
Airflow.
"""

from __future__ import annotations

import os
from typing import Dict

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


DEFAULT_POSTGRES_PACKAGE = "org.postgresql:postgresql:42.7.1"


def build_spark_session() -> SparkSession:
    """Create a :class:`SparkSession` configured for JDBC access.

    The function honors the following environment variables:

    ``SPARK_APP_NAME``
        Custom application name. Defaults to ``"video-app-kpi"``.
    ``SPARK_JARS_PACKAGES``
        Comma separated list of extra packages to load.
    ``SPARK_EXTRA_JARS``
        Optional path to custom JDBC jars available on the workers.
    ``SPARK_SQL_TIMEZONE``
        Time zone used for date truncation/aggregation (defaults to ``UTC``).
    """

    app_name = os.getenv("SPARK_APP_NAME", "video-app-kpi")
    builder = SparkSession.builder.appName(app_name)

    packages = os.getenv("SPARK_JARS_PACKAGES", DEFAULT_POSTGRES_PACKAGE)
    if packages:
        builder = builder.config("spark.jars.packages", packages)

    extra_jars = os.getenv("SPARK_EXTRA_JARS")
    if extra_jars:
        builder = builder.config("spark.jars", extra_jars)

    timezone = os.getenv("SPARK_SQL_TIMEZONE", "UTC")
    builder = builder.config("spark.sql.session.timeZone", timezone)

    spark = builder.getOrCreate()

    log_level = os.getenv("SPARK_LOG_LEVEL")
    if log_level:
        spark.sparkContext.setLogLevel(log_level)

    return spark


def build_jdbc_options() -> Dict[str, str]:
    """Return JDBC options used by Spark to connect to PostgreSQL."""

    url = os.getenv("SPARK_JDBC_URL")
    if not url:
        host = os.getenv("SUPABASE_DB_HOST", os.getenv("DB_HOST", "localhost"))
        port = os.getenv("SUPABASE_DB_PORT", os.getenv("DB_PORT", "5432"))
        database = os.getenv("SUPABASE_DB_NAME", os.getenv("DB_NAME", "postgres"))
        url = f"jdbc:postgresql://{host}:{port}/{database}"

    user = os.getenv("SUPABASE_DB_USER", os.getenv("DB_USER", "postgres"))
    password = os.getenv("SUPABASE_DB_PASSWORD", os.getenv("DB_PASSWORD"))
    driver = os.getenv("SPARK_JDBC_DRIVER", "org.postgresql.Driver")

    options: Dict[str, str] = {
        "url": url,
        "user": user,
        "driver": driver,
    }
    if password:
        options["password"] = password

    return options


def load_jobs_dataframe(spark: SparkSession) -> DataFrame:
    """Load the ``jobs`` table from PostgreSQL using Spark JDBC."""

    jdbc_options = build_jdbc_options()
    table = os.getenv("SPARK_JOBS_TABLE", "jobs")

    return (
        spark.read.format("jdbc")
        .options(**jdbc_options)
        .option("dbtable", table)
        .load()
    )


def compute_daily_kpis(jobs_df: DataFrame) -> DataFrame:
    """Compute aggregated metrics per day for the admin dashboard."""

    columns = set(jobs_df.columns)

    if {"finished_at", "started_at"}.issubset(columns):
        jobs_df = jobs_df.withColumn(
            "duration_seconds",
            F.col("finished_at").cast("long") - F.col("started_at").cast("long"),
        )
    else:
        jobs_df = jobs_df.withColumn("duration_seconds", F.lit(None).cast("double"))

    jobs_df = jobs_df.withColumn(
        "duration_seconds",
        F.when(F.col("duration_seconds") >= 0, F.col("duration_seconds")).otherwise(None),
    ).withColumn("duration_seconds", F.col("duration_seconds").cast("double"))

    date_expr = F.col("created_at") if "created_at" in columns else F.current_timestamp()
    if "finished_at" in columns:
        date_expr = F.coalesce(F.col("finished_at"), date_expr)

    jobs_df = jobs_df.withColumn("job_date", F.to_date(F.date_trunc("day", date_expr)))

    status_col = F.lower(F.col("status")) if "status" in columns else F.lit(None)

    active_users_expr = (
        F.countDistinct("user_id")
        if "user_id" in columns
        else F.first(F.lit(0))
    )

    aggregated = (
        jobs_df.groupBy("job_date")
        .agg(
            F.count(F.lit(1)).alias("jobs_total"),
            F.count(F.when(status_col == "succeeded", True)).alias(
                "jobs_succeeded"
            ),
            F.count(F.when(status_col == "failed", True)).alias("jobs_failed"),
            active_users_expr.alias("active_users"),
            F.avg("duration_seconds").alias("avg_duration_seconds"),
            F.expr("percentile_approx(duration_seconds, 0.5)").alias(
                "p50_duration_seconds"
            ),
        )
        .orderBy("job_date")
    )

    aggregated = aggregated.filter(F.col("job_date").isNotNull())

    aggregated = aggregated.withColumn(
        "success_rate",
        F.when(
            F.col("jobs_total") > 0,
            F.col("jobs_succeeded") / F.col("jobs_total"),
        ),
    )

    return (
        aggregated.withColumn(
            "avg_duration_seconds",
            F.round("avg_duration_seconds", 2).cast("double"),
        )
        .withColumn(
            "p50_duration_seconds", F.round("p50_duration_seconds", 2).cast("double")
        )
        .withColumn("success_rate", F.round("success_rate", 4).cast("double"))
        .withColumn("updated_at", F.current_timestamp())
    )


def write_kpis(kpis_df: DataFrame) -> None:
    """Persist the KPI dataframe either to PostgreSQL or to storage."""

    output_path = os.getenv("SPARK_KPI_OUTPUT_PATH")
    if output_path:
        output_format = os.getenv("SPARK_KPI_OUTPUT_FORMAT", "parquet")
        writer = kpis_df.write.mode(os.getenv("SPARK_KPI_WRITE_MODE", "overwrite"))
        if output_format == "csv":
            writer = writer.option("header", "true")
        writer.format(output_format).save(output_path)
        return

    jdbc_options = build_jdbc_options()
    table = os.getenv("SPARK_KPI_TABLE", "kpi_spark_daily")
    writer = (
        kpis_df.write.format("jdbc")
        .options(**jdbc_options)
        .option("dbtable", table)
        .mode(os.getenv("SPARK_KPI_WRITE_MODE", "overwrite"))
    )
    if os.getenv("SPARK_KPI_TRUNCATE", "true").lower() == "true":
        writer = writer.option("truncate", "true")
    writer.save()


def main() -> None:
    """Entrypoint used by ``spark-submit``."""

    spark = build_spark_session()
    try:
        jobs_df = load_jobs_dataframe(spark)
        if jobs_df.rdd.isEmpty():
            print("No jobs found in the source table; skipping KPI computation.")
            return

        kpis_df = compute_daily_kpis(jobs_df)
        print("Writing Spark KPIs ...")
        write_kpis(kpis_df)
        print("Spark KPI pipeline finished successfully.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
