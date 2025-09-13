from pyspark.sql import SparkSession

def main():
    spark = SparkSession.builder.appName("video-log-analysis").getOrCreate()
    logs = spark.read.json("s3a://logs/videos/")
    stats = logs.groupBy("status").count()
    stats.show()
    spark.stop()


if __name__ == "__main__":
    main()
