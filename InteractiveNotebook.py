# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %md
# MAGIC # MLflow Regression Pipeline Databricks Notebook
# MAGIC This notebook runs the MLflow Regression Pipeline on Databricks and inspects its results.
# MAGIC 
# MAGIC For more information about the MLflow Regression Pipeline, including usage examples,
# MAGIC see the [Regression Pipeline overview documentation](https://mlflow.org/docs/latest/pipelines.html#regression-pipeline)
# MAGIC and the [Regression Pipeline API documentation](https://mlflow.org/docs/latest/python_api/mlflow.pipelines.html#module-mlflow.pipelines.regression.v1.pipeline).

# COMMAND ----------

# MAGIC %pip install -U mlflow-1.26.2.dev0-py3-none-any.whl[pipelines] pandas-profiling

# COMMAND ----------

from mlflow.pipelines import Pipeline

p = Pipeline(profile="databricks")

# COMMAND ----------

p.clean()

# COMMAND ----------

p.inspect()

# COMMAND ----------

p.run("ingest")

# COMMAND ----------

p.run("split")

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from mlflow.pipelines.decorators import transformer_method


@transformer_method
def transform_method():

    return Pipeline(
        steps=[
            (
                "encoder",
                ColumnTransformer(
                    transformers=[
                        (
                            "hour_encoder",
                            OneHotEncoder(categories="auto", sparse=False),
                            ["pickup_hour", "pickup_dow"],
                        ),
                        (
                            "minmax_scaler",
                            MinMaxScaler(),
                            ["trip_distance", "trip_duration"],
                        ),
                    ]
                ),
            ),
        ]
    )

# COMMAND ----------

p.run("transform")

# COMMAND ----------

import sklearn.linear_model
from mlflow.pipelines.decorators import estimator_method


@estimator_method
def custom_regressor():
    #return sklearn.linear_model.LinearRegression()
    return sklearn.linear_model.ElasticNet()

# COMMAND ----------

p.run("train")

# COMMAND ----------

p.run("evaluate")

# COMMAND ----------

p.run("register")

# COMMAND ----------

p.inspect("train")

# COMMAND ----------

test_data = p.get_artifact("test_data")
test_data.describe()

# COMMAND ----------

trained_model = p.get_artifact("model")
print(trained_model)

# COMMAND ----------

# MAGIC %fs ls dbfs:/databricks-datasets

# COMMAND ----------


