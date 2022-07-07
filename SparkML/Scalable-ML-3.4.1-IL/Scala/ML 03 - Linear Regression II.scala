// Databricks notebook source
// MAGIC 
// MAGIC %md-sandbox
// MAGIC 
// MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
// MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
// MAGIC </div>

// COMMAND ----------

// MAGIC %md
// MAGIC # Linear Regression: Improving our model
// MAGIC 
// MAGIC In this notebook we will be adding additional features to our model, as well as discuss how to handle categorical features.
// MAGIC 
// MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
// MAGIC  - One Hot Encode categorical variables
// MAGIC  - Use the Pipeline API
// MAGIC  - Save and load models

// COMMAND ----------

// MAGIC %run "./Includes/Classroom-Setup"

// COMMAND ----------

val filePath = "dbfs:/mnt/training/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
val airbnbDF = spark.read.format("delta").load(filePath)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Train/Test Split
// MAGIC 
// MAGIC Let's use the same 80/20 split with the same seed as the previous notebook so we can compare our results apples to apples (unless you changed the cluster config!)

// COMMAND ----------

val Array(trainDF, testDF) = airbnbDF.randomSplit(Array(.8, .2), seed=42)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Categorical Variables
// MAGIC 
// MAGIC There are a few ways to handle categorical features:
// MAGIC * Assign them a numeric value
// MAGIC * Create "dummy" variables (also known as One Hot Encoding)
// MAGIC * Generate embeddings (mainly used for textual data)
// MAGIC 
// MAGIC ### One Hot Encoder
// MAGIC Here, we are going to One Hot Encode (OHE) our categorical variables. Spark doesn't have a `dummies` function, and OHE is a two step process. First, we need to use `StringIndexer` to map a string column of labels to an ML column of label indices [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.StringIndexer)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.StringIndexer).
// MAGIC 
// MAGIC Then, we can apply the `OneHotEncoder` to the output of the StringIndexer [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.OneHotEncoder)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.OneHotEncoder).

// COMMAND ----------

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

val categoricalCols = trainDF.dtypes.filter(_._2 == "StringType").map(_._1)
val indexOutputCols = categoricalCols.map(_ + "Index")
val oheOutputCols = categoricalCols.map(_ + "OHE")

val stringIndexer = new StringIndexer()
  .setInputCols(categoricalCols)
  .setOutputCols(indexOutputCols)
  .setHandleInvalid("skip")

val oheEncoder = new OneHotEncoder()
  .setInputCols(indexOutputCols)
  .setOutputCols(oheOutputCols)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Vector Assembler
// MAGIC 
// MAGIC Now we can combine our OHE categorical features with our numeric features.

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler

val numericCols = trainDF.dtypes.filter{ case (field, dataType) => dataType == "DoubleType" && field != "price"}.map(_._1)
val assemblerInputs = oheOutputCols ++ numericCols
val vecAssembler = new VectorAssembler()
  .setInputCols(assemblerInputs)
  .setOutputCol("features")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Linear Regression
// MAGIC 
// MAGIC Now that we have all of our features, let's build a linear regression model.

// COMMAND ----------

import org.apache.spark.ml.regression.LinearRegression

val lr = new LinearRegression()
  .setLabelCol("price")
  .setFeaturesCol("features")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Pipeline
// MAGIC 
// MAGIC Let's put all these stages in a Pipeline. A `Pipeline` is a way of organizing all of our transformers and estimators [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.Pipeline)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.Pipeline).
// MAGIC 
// MAGIC This way, we don't have to worry about remembering the same ordering of transformations to apply to our test dataset.

// COMMAND ----------

import org.apache.spark.ml.Pipeline

val stages = Array(stringIndexer, oheEncoder, vecAssembler,  lr)

val pipeline = new Pipeline()
  .setStages(stages)

val pipelineModel = pipeline.fit(trainDF)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Saving Models
// MAGIC 
// MAGIC We can save our models to persistent storage (e.g. DBFS) in case our cluster goes down so we don't have to recompute our results.

// COMMAND ----------

val pipelinePath = userhome + "/machine-learning-s/lr_pipeline_model"
pipelineModel.write.overwrite().save(pipelinePath)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Loading models
// MAGIC 
// MAGIC When you load in models, you need to know the type of model you are loading back in (was it a linear regression or logistic regression model?).
// MAGIC 
// MAGIC For this reason, we recommend you always put your transformers/estimators into a Pipeline, so you can always load the generic PipelineModel back in.

// COMMAND ----------

import org.apache.spark.ml.PipelineModel

val savedPipelineModel = PipelineModel.load(pipelinePath)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Apply model to test set

// COMMAND ----------

val predDF = savedPipelineModel.transform(testDF)

display(predDF.select("features", "price", "prediction"))

// COMMAND ----------

// MAGIC %md
// MAGIC ## Evaluate model
// MAGIC 
// MAGIC ![](https://files.training.databricks.com/images/r2d2.jpg) How is our R2 doing? 

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator

val regressionEvaluator = new RegressionEvaluator().setPredictionCol("prediction").setLabelCol("price").setMetricName("rmse")

val rmse = regressionEvaluator.evaluate(predDF)
val r2 = regressionEvaluator.setMetricName("r2").evaluate(predDF)
println(s"RMSE is $rmse")
println(s"R2 is $r2")
println("*-"*80)

// COMMAND ----------

// MAGIC %md
// MAGIC As you can see, our RMSE decreased when compared to the model without one-hot encoding, and the R2 increased as well!

// COMMAND ----------

// MAGIC %md-sandbox
// MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
// MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
// MAGIC <br/>
// MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
