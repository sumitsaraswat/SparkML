// Databricks notebook source
// MAGIC 
// MAGIC %md-sandbox
// MAGIC 
// MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
// MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
// MAGIC </div>

// COMMAND ----------

// MAGIC %md
// MAGIC # Random Forests and Hyperparameter Tuning
// MAGIC 
// MAGIC Now let's take a look at how to tune random forests using grid search and cross validation in order to find the optimal hyperparameters.  Using the Databricks Runtime for ML, MLflow automatically logs your experiments with the SparkML cross-validator!
// MAGIC 
// MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
// MAGIC  - Tune hyperparameters using Grid Search
// MAGIC  - Optimize a SparkML pipeline

// COMMAND ----------

// MAGIC %run "./Includes/Classroom-Setup"

// COMMAND ----------

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.Pipeline

val filePath = "dbfs:/mnt/training/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
val airbnbDF = spark.read.format("delta").load(filePath)
val Array(trainDF, testDF) = airbnbDF.randomSplit(Array(.8, .2), seed=42)

val categoricalCols = trainDF.dtypes.filter(_._2 == "StringType").map(_._1)
val indexOutputCols = categoricalCols.map(_ + "Index")

val stringIndexer = new StringIndexer()
  .setInputCols(categoricalCols)
  .setOutputCols(indexOutputCols)
  .setHandleInvalid("skip")

val numericCols = trainDF.dtypes.filter{ case (field, dataType) => dataType == "DoubleType" && field != "price"}.map(_._1)
val assemblerInputs = indexOutputCols ++ numericCols
val vecAssembler = new VectorAssembler()
  .setInputCols(assemblerInputs)
  .setOutputCol("features")

val rf = new RandomForestRegressor()
  .setLabelCol("price")
  .setMaxBins(40)

val stages = Array(stringIndexer, vecAssembler, rf)
val pipeline = new Pipeline()
  .setStages(stages)

// COMMAND ----------

// MAGIC %md
// MAGIC ## ParamGrid
// MAGIC 
// MAGIC First let's take a look at the various hyperparameters we could tune for random forest.
// MAGIC 
// MAGIC **Pop quiz:** what's the difference between a parameter and a hyperparameter?

// COMMAND ----------

rf.explainParams

// COMMAND ----------

// MAGIC %md
// MAGIC There are a lot of hyperparameters we could tune, and it would take a long time to manually configure.
// MAGIC 
// MAGIC Instead of a manual (ad-hoc) approach, let's use Spark's `ParamGridBuilder` to find the optimal hyperparameters in a more systematic approach [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.ParamGridBuilder)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.tuning.ParamGridBuilder).
// MAGIC 
// MAGIC Let's define a grid of hyperparameters to test:
// MAGIC   - `maxDepth`: max depth of each decision tree (Use the values `2, 5`)
// MAGIC   - `numTrees`: number of decision trees to train (Use the values `5, 10`)
// MAGIC 
// MAGIC `addGrid()` accepts the name of the parameter (e.g. `rf.maxDepth`), and a list of the possible values (e.g. `[2, 5]`).

// COMMAND ----------

import org.apache.spark.ml.tuning.ParamGridBuilder

val paramGrid = new ParamGridBuilder()
  .addGrid(rf.maxDepth, Array(2, 5))
  .addGrid(rf.numTrees, Array(5, 10))
  .build()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Cross Validation
// MAGIC 
// MAGIC We are also going to use 3-fold cross validation to identify the optimal hyperparameters.
// MAGIC 
// MAGIC ![crossValidation](https://files.training.databricks.com/images/301/CrossValidation.png)
// MAGIC 
// MAGIC With 3-fold cross-validation, we train on 2/3 of the data, and evaluate with the remaining (held-out) 1/3. We repeat this process 3 times, so each fold gets the chance to act as the validation set. We then average the results of the three rounds.

// COMMAND ----------

// MAGIC %md
// MAGIC We pass in the `estimator` (pipeline), `evaluator`, and `estimatorParamMaps` to `CrossValidator` so that it knows:
// MAGIC - Which model to use
// MAGIC - How to evaluate the model
// MAGIC - What hyperparameters to set for the model
// MAGIC 
// MAGIC We can also set the number of folds we want to split our data into (3), as well as setting a seed so we all have the same split in the data [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.tuning.CrossValidator).

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.CrossValidator

val evaluator = new RegressionEvaluator()
  .setLabelCol("price")
  .setPredictionCol("prediction")

val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)
  .setSeed(42)

// COMMAND ----------

// MAGIC %md
// MAGIC **Question**: How many models are we training right now?

// COMMAND ----------

val cvModel = cv.fit(trainDF)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Parallelism Parameter
// MAGIC 
// MAGIC Hmmm... that took a long time to run. That's because the models were being trained sequentially rather than in parallel!
// MAGIC 
// MAGIC In Spark 2.3, a [parallelism](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator.parallelism) parameter was introduced. From the docs: `the number of threads to use when running parallel algorithms (>= 1)`.
// MAGIC 
// MAGIC Let's set this value to 4 and see if we can train any faster. The Spark [docs](https://spark.apache.org/docs/latest/ml-tuning.html) recommend a value between 2-10.

// COMMAND ----------

val cvModel = cv.setParallelism(4).fit(trainDF)

// COMMAND ----------

// MAGIC %md
// MAGIC **Question**: Hmmm... that still took a long time to run. Should we put the pipeline in the cross validator, or the cross validator in the pipeline?
// MAGIC 
// MAGIC It depends if there are estimators or transformers in the pipeline. If you have things like StringIndexer (an estimator) in the pipeline, then you have to refit it every time if you put the entire pipeline in the cross validator.
// MAGIC 
// MAGIC However, if there is any concern about data leakage from the earlier steps, the safest thing is to put the pipeline inside the CV, not the other way. CV first splits the data and then .fit() the pipeline. If it is placed at the end of the pipeline, we potentially can leak the info from hold-out set to train set.

// COMMAND ----------

val cv = new CrossValidator()
  .setEstimator(rf)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)
  .setParallelism(4)
  .setSeed(42)

val stagesWithCV = Array(stringIndexer, vecAssembler, cv)

val pipeline = new Pipeline()
  .setStages(stagesWithCV)

val pipelineModel = pipeline.fit(trainDF)

// COMMAND ----------

// MAGIC %md
// MAGIC Let's take a look at the model with the best hyperparameter configuration

// COMMAND ----------

cvModel.getEstimatorParamMaps.zip(cvModel.avgMetrics)

// COMMAND ----------

val predDF = pipelineModel.transform(testDF)

val rmse = evaluator.evaluate(predDF)
val r2 = evaluator.setMetricName("r2").evaluate(predDF)
println(s"RMSE is $rmse")
println(s"R2 is $r2")
println("*-"*80)

// COMMAND ----------

// MAGIC %md
// MAGIC Progress!  Looks like we're out-performing decision trees.

// COMMAND ----------

// MAGIC %md-sandbox
// MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
// MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
// MAGIC <br/>
// MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
