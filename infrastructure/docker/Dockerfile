FROM python:3.11-slim

# Build args
ARG SPARK_VERSION=3.5.1
ARG HADOOP_VERSION=3

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_HOME=/opt/spark
ENV PATH="${PATH}:${SPARK_HOME}/bin:${SPARK_HOME}/sbin"
ENV PYTHONPATH="${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.10.9.7-src.zip:${PYTHONPATH}"
ENV PYSPARK_PYTHON=python3

# Install Java and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk-headless \
    curl \
    wget \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install Spark
RUN wget -q "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
    && tar -xzf "spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" -C /opt \
    && mv "/opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" /opt/spark \
    && rm "spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz"

# Add Delta Lake JAR
RUN wget -q "https://repo1.maven.org/maven2/io/delta/delta-core_2.12/2.4.0/delta-core_2.12-2.4.0.jar" \
    -O "${SPARK_HOME}/jars/delta-core_2.12-2.4.0.jar"

# Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Airflow setup
ENV AIRFLOW_HOME=/app/airflow
RUN airflow db init && \
    airflow users create \
        --username admin \
        --password admin \
        --firstname Preet \
        --lastname Shah \
        --role Admin \
        --email shahpreetp15@gmail.com

EXPOSE 8080

CMD ["airflow", "webserver", "--port", "8080"]
