// Databricks notebook source
// MAGIC 
// MAGIC %md-sandbox
// MAGIC 
// MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
// MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
// MAGIC </div>

// COMMAND ----------

// MAGIC %md
// MAGIC # Hyperparameter Tuning with Random Forests
// MAGIC 
// MAGIC In this lab, you will convert the Airbnb problem to a classification dataset, build a random forest classifier, and tune some hyperparameters of the random forest.
// MAGIC 
// MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
// MAGIC  - Perform grid search on a random forest
// MAGIC  - Get the feature importances across the forest
// MAGIC  - Save the model
// MAGIC  - Identify differences between scikit-learn's Random Forest and SparkML's
// MAGIC  
// MAGIC You can read more about the distributed implementation of Random Forests in the Spark [source code](https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/tree/impl/RandomForest.scala#L42).

// COMMAND ----------

// MAGIC %run "../Includes/Classroom-Setup"

// COMMAND ----------

// MAGIC %md
// MAGIC ## From Regression to Classification
// MAGIC 
// MAGIC In this case, we'll turn the Airbnb housing dataset into a classification problem to **classify between high and low price listings.**  Our `class` column will be:<br><br>
// MAGIC 
// MAGIC - `0` for a low cost listing of under $150
// MAGIC - `1` for a high cost listing of $150 or more

// COMMAND ----------

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline

val filePath = "dbfs:/mnt/training/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"

val airbnbDF = spark.read.format("delta").load(filePath)
  .withColumn("priceClass", ($"price" >= 150).cast("int"))
  .drop($"price")

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

// COMMAND ----------

// MAGIC %md
// MAGIC ## Why can't we OHE?
// MAGIC 
// MAGIC **Question:** What would go wrong if we One Hot Encoded our variables before passing them into the random forest?
// MAGIC 
// MAGIC **HINT:** Think about what would happen to the "randomness" of feature selection.

// COMMAND ----------

// MAGIC %md
// MAGIC ## Random Forest
// MAGIC 
// MAGIC Create a Random Forest classifer called `rf` with the `labelCol`=`priceClass`, `maxBins`=`40`, and `seed`=`42` (for reproducibility).
// MAGIC 
// MAGIC It's under `pyspark.ml.classification.RandomForestClassifier` in Python and `org.apache.spark.ml.classification.RandomForestClassifier` in Scala.

// COMMAND ----------

// TODO

val rf = <FILL_IN>

// COMMAND ----------

// MAGIC %md
// MAGIC ## Grid Search
// MAGIC 
// MAGIC There are a lot of hyperparameters we could tune, and it would take a long time to manually configure.
// MAGIC 
// MAGIC Let's use Spark's `ParamGridBuilder` to find the optimal hyperparameters in a more systematic approach [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.ParamGridBuilder)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.tuning.ParamGridBuilder).
// MAGIC 
// MAGIC Let's define a grid of hyperparameters to test:
// MAGIC   - maxDepth: max depth of the decision tree (Use the values `2, 5, 10`)
// MAGIC   - numTrees: number of decision trees (Use the values `10, 20, 100`)
// MAGIC 
// MAGIC `addGrid()` accepts the name of the parameter (e.g. `rf.maxDepth`), and a list of the possible values (e.g. `[2, 5, 10]`).

// COMMAND ----------

// TODO

// COMMAND ----------

// MAGIC %md-sandbox
// MAGIC ## Evaluator
// MAGIC 
// MAGIC In the past, we used a `RegressionEvaluator`.  For classification, we can use a [BinaryClassificationEvaluator](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.evaluation.BinaryClassificationEvaluator) if we have two classes or [MulticlassClassificationEvaluator](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.evaluation.MulticlassClassificationEvaluator) for more than two classes.
// MAGIC 
// MAGIC Create a `BinaryClassificationEvaluator` with `areaUnderROC` as the metric.
// MAGIC 
// MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> [Read more on ROC curves here.](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)  In essence, it compares true positive and false positives.

// COMMAND ----------

// TODO

// COMMAND ----------

// MAGIC %md
// MAGIC ## Cross Validation
// MAGIC 
// MAGIC We are going to do 3-Fold cross-validation, with `parallelism`=4, and set the `seed`=42 on the cross-validator for reproducibility.
// MAGIC 
// MAGIC Put the Random Forest in the CV to speed up the cross validation (as opposed to the pipeline in the CV) [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.tuning.CrossValidator).

// COMMAND ----------

// TODO
import org.apache.spark.ml.tuning.CrossValidator

val cv = <FILL_IN>

// COMMAND ----------

// MAGIC %md
// MAGIC ## Pipeline
// MAGIC 
// MAGIC Let's fit the pipeline with our cross validator to our training data (this may take a few minutes).

// COMMAND ----------

import org.apache.spark.ml.Pipeline

val stages = Array(stringIndexer, vecAssembler, cv)

val pipeline = new Pipeline()
  .setStages(stages)

val pipelineModel = pipeline.fit(trainDF)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Hyperparameter
// MAGIC 
// MAGIC Which hyperparameter combination performed the best?

// COMMAND ----------

val cvModel = pipelineModel.stages.last.asInstanceOf[org.apache.spark.ml.tuning.CrossValidatorModel]
val rfModel = cvModel.bestModel

// cvModel.getEstimatorParamMaps.zip(cvModel.avgMetrics)

println(rfModel.explainParams())

// COMMAND ----------

// MAGIC %md
// MAGIC ## Feature Importance

// COMMAND ----------

val featureImportances = rfModel.asInstanceOf[org.apache.spark.ml.classification.RandomForestClassificationModel].featureImportances
val data = vecAssembler.getInputCols.zip(featureImportances.toArray)
val columns = Array("feature", "importance")
val topFeatures = spark.createDataFrame(data).toDF(columns: _*).orderBy($"importance".desc)
display(topFeatures)

// COMMAND ----------

// MAGIC %md
// MAGIC Do those features make sense? Would you use those features when picking an Airbnb rental?

// COMMAND ----------

// MAGIC %md
// MAGIC ## Apply Model to test set

// COMMAND ----------

// TODO
val predDF = <FILL_IN>
val areaUnderROC = <FILL_IN>

println(s"Area under ROC is $areaUnderROC")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Save Model
// MAGIC 
// MAGIC Save the model to `<userhome>/rf_pipeline_model`.

// COMMAND ----------

// TODO

// COMMAND ----------

// MAGIC %md
// MAGIC ## Sklearn vs SparkML
// MAGIC 
// MAGIC [Sklearn RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) vs `SparkML RandomForestRegressor` [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.RandomForestRegressor)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.regression.RandomForestRegressor).
// MAGIC 
// MAGIC Look at these params in particular:
// MAGIC * **n_estimators** (sklearn) vs **numTrees** (SparkML)
// MAGIC * **max_depth** (sklearn) vs **maxDepth** (SparkML)
// MAGIC * **max_features** (sklearn) vs **featureSubsetStrategy** (SparkML)
// MAGIC * **maxBins** (SparkML only)
// MAGIC 
// MAGIC What do you notice that is different?

// COMMAND ----------

// MAGIC %md-sandbox
// MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
// MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
// MAGIC <br/>
// MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
