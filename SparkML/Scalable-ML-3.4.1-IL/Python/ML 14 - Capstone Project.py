# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Capstone Project
# MAGIC 
# MAGIC You will work in small teams (2-3 people) to do the following:
# MAGIC 
# MAGIC 0. Load in a dataset from `databricks-datasets`, [UCI ML Repository](https://archive.ics.uci.edu/ml/index.php), [Kaggle](https://www.kaggle.com/), or any other open-source dataset. 
# MAGIC 0. Create a Delta Table
# MAGIC 0. Build either:
# MAGIC   * Sklearn model, but apply it in parallel using a Pandas UDF
# MAGIC   * Build an MLlib model
# MAGIC 0. Track model performance with MLflow
# MAGIC 0. Present your notebook to the class and share any roadblocks you hit (~5 minutes)
# MAGIC 
# MAGIC Extras:
# MAGIC * Build a Databricks Dashboard
# MAGIC * Perform hyperparameter tuning
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Work in a small team to apply the skills you learned throughout the course to a new dataset

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
