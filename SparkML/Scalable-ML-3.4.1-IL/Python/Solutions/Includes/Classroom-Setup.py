# Databricks notebook source
# MAGIC 
# MAGIC %python
# MAGIC course_name = "Machine-learning"

# COMMAND ----------

# MAGIC %run "./Dataset-Mounts"

# COMMAND ----------

# MAGIC %python
# MAGIC dbutils.fs.mkdirs("dbfs:/user/" + username)
# MAGIC dbutils.fs.mkdirs("dbfs:/user/" + username + "/machine-learning-p")
# MAGIC dbutils.fs.mkdirs("dbfs:/user/" + username + "/machine-learning-s")
# MAGIC None

# COMMAND ----------

# MAGIC %python
# MAGIC # This is needed for the Delta & MLflow model registry notebook & lab
# MAGIC import re
# MAGIC 
# MAGIC split_username = username.split("@")[0].replace("-", "_")
# MAGIC cleaned_username = re.sub(r'\W+', '', split_username)
# MAGIC spark.conf.set("cleaned_username", cleaned_username)

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC def untilStreamIsReady(name):
# MAGIC   queries = list(filter(lambda query: query.name == name, spark.streams.active))
# MAGIC 
# MAGIC   if len(queries) == 0:
# MAGIC     print("The stream is not active.")
# MAGIC 
# MAGIC   else:
# MAGIC     while (queries[0].isActive and len(queries[0].recentProgress) == 0):
# MAGIC       pass # wait until there is any type of progress
# MAGIC 
# MAGIC     if queries[0].isActive:
# MAGIC       queries[0].awaitTermination(5)
# MAGIC       print("The stream is active and ready.")
# MAGIC     else:
# MAGIC       print("The stream is not active.")
# MAGIC 
# MAGIC None

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC val cleaned_username = spark.conf.get("cleaned_username")
# MAGIC 
# MAGIC def untilStreamIsReady(name:String):Unit = {
# MAGIC   val queries = spark.streams.active.filter(_.name == name)
# MAGIC 
# MAGIC   if (queries.length == 0) {
# MAGIC     println("The stream is not active.")
# MAGIC   } else {
# MAGIC     while (queries(0).isActive && queries(0).recentProgress.length == 0) {
# MAGIC       // wait until there is any type of progress
# MAGIC     }
# MAGIC 
# MAGIC     if (queries(0).isActive) {
# MAGIC       queries(0).awaitTermination(5*1000)
# MAGIC       println("The stream is active and ready.")
# MAGIC     } else {
# MAGIC       println("The stream is not active.")
# MAGIC     }
# MAGIC   }
# MAGIC }
# MAGIC 
# MAGIC displayHTML("""
# MAGIC <div>Declared various utility methods:</div>
# MAGIC <li>Declared <b style="color:green">untilStreamIsReady(<i>name:String</i>)</b> to control workflow</li>
# MAGIC <br/>
# MAGIC <div>All done!</div>
# MAGIC """)
