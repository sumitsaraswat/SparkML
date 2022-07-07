// Databricks notebook source
// MAGIC 
// MAGIC %md-sandbox
// MAGIC 
// MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
// MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
// MAGIC </div>

// COMMAND ----------

// MAGIC %md
// MAGIC ![](https://files.training.databricks.com/images/301/deployment_options_mllib.png)
// MAGIC 
// MAGIC There are four main deployment options:
// MAGIC * Batch pre-compute
// MAGIC * Structured streaming
// MAGIC * Low-latency model serving
// MAGIC * Mobile/embedded (outside scope of class)
// MAGIC 
// MAGIC We have already seen how to do batch predictions using Spark. Now let's look at how to make predictions on streaming data.
// MAGIC 
// MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
// MAGIC  - Apply a SparkML model on a simulated stream of data 

// COMMAND ----------

// MAGIC %run "./Includes/Classroom-Setup"

// COMMAND ----------

// MAGIC %md
// MAGIC ## Load in Model & Data
// MAGIC 
// MAGIC We are loading in a repartitioned version of our dataset (100 partitions instead of 4) to see more incremental progress of the streaming predictions.

// COMMAND ----------

import org.apache.spark.ml.PipelineModel

val pipelinePath = "dbfs:/mnt/training/airbnb/sf-listings/models/sf-listings-2019-03-06/pipeline_model"
val pipelineModel = PipelineModel.load(pipelinePath)

val repartitionedPath =  "dbfs:/mnt/training/airbnb/sf-listings/sf-listings-2019-03-06-clean-100p.parquet/"
val schema = spark.read.parquet(repartitionedPath).schema

// COMMAND ----------

// MAGIC %md
// MAGIC ## Simulate streaming data
// MAGIC 
// MAGIC **NOTE**: You must specify a schema when creating a streaming source DataFrame.

// COMMAND ----------

val streamingData = spark
  .readStream
  .schema(schema) // Can set the schema this way
  .option("maxFilesPerTrigger", 1)
  .parquet(repartitionedPath)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Make Predictions

// COMMAND ----------

val streamPred = pipelineModel.transform(streamingData)

// COMMAND ----------

// MAGIC %md
// MAGIC Let's save our results.

// COMMAND ----------

val checkpointDir = userhome + "/machine-learning/pred_stream_1s_checkpoint"
// Clear out the checkpointing directory
dbutils.fs.rm(checkpointDir, true) 

streamPred
 .writeStream
 .format("memory")
 .option("checkpointLocation", checkpointDir)
 .outputMode("append")
 .queryName("pred_stream_1s")
 .start()

// COMMAND ----------

untilStreamIsReady("pred_stream_1s")

// COMMAND ----------

// MAGIC %md
// MAGIC While this is running, take a look at the new Structured Streaming tab in the Spark UI.

// COMMAND ----------

display(
  sql("select * from pred_stream_1s")
)

// COMMAND ----------

display(
  sql("select count(*) from pred_stream_1s")
)

// COMMAND ----------

// MAGIC %md
// MAGIC Now that we are done, make sure to stop the stream

// COMMAND ----------

for (stream <- spark.streams.active) {
  println("Stopping " + stream.name)
  stream.stop() // Stop the stream
}

// COMMAND ----------

// MAGIC %md
// MAGIC ### What about Model Export?
// MAGIC 
// MAGIC * [MLeap](https://mleap-docs.combust.ml/)
// MAGIC   * Company that developed MLeap is no longer supporting it, and MLeap does not yet support Scala 2.12/Spark 3.0
// MAGIC * [ONNX](https://onnx.ai/)
// MAGIC   * ONNX is very popular in the deep learning community allowing developers to switch between libraries and languages, but only has experimental support for MLlib.
// MAGIC * DIY (Reimplement it yourself)
// MAGIC   * Error-prone, fragile
// MAGIC * 3rd party libraries
// MAGIC   * See XGBoost notebook
// MAGIC   * [H2O](https://www.h2o.ai/products/h2o-sparkling-water/)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Low-Latency Serving Solutions
// MAGIC 
// MAGIC Low-latency serving can operate as quickly as tens to hundreds of milliseconds.  Custom solutions are normally backed by Docker and/or Flask (though Flask generally isn't recommended in production unless significant precations are taken).  Managed solutions also include:<br><br>
// MAGIC 
// MAGIC * [MLflow Model Serving (Preview)](https://databricks.com/blog/2020/06/25/announcing-mlflow-model-serving-on-databricks.html)
// MAGIC * [Azure Machine Learning](https://azure.microsoft.com/en-us/services/machine-learning/)
// MAGIC * [SageMaker](https://aws.amazon.com/sagemaker/)

// COMMAND ----------

// MAGIC %md-sandbox
// MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
// MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
// MAGIC <br/>
// MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
