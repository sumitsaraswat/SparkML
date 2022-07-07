# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Registry
# MAGIC 
# MAGIC MLflow Model Registry is a collaborative hub where teams can share ML models, work together from experimentation to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance.  This lesson explores how to manage models using the MLflow model registry.
# MAGIC 
# MAGIC This demo notebook will use scikit-learn on the Airbnb dataset, but in the lab you will use MLlib.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Register a model using MLflow
# MAGIC  - Deploy that model into production
# MAGIC  - Update a model in production to new version including a staging phase for testing
# MAGIC  - Archive and delete models

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Model Registry
# MAGIC 
# MAGIC The MLflow Model Registry component is a centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of an MLflow Model. It provides model lineage (which MLflow Experiment and Run produced the model), model versioning, stage transitions (e.g. from staging to production), annotations (e.g. with comments, tags), and deployment management (e.g. which production jobs have requested a specific model version).
# MAGIC 
# MAGIC Model registry has the following features:<br><br>
# MAGIC 
# MAGIC * **Central Repository:** Register MLflow models with the MLflow Model Registry. A registered model has a unique name, version, stage, and other metadata.
# MAGIC * **Model Versioning:** Automatically keep track of versions for registered models when updated.
# MAGIC * **Model Stage:** Assigned preset or custom stages to each model version, like “Staging” and “Production” to represent the lifecycle of a model.
# MAGIC * **Model Stage Transitions:** Record new registration events or changes as activities that automatically log users, changes, and additional metadata such as comments.
# MAGIC * **CI/CD Workflow Integration:** Record stage transitions, request, review and approve changes as part of CI/CD pipelines for better control and governance.
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/model-registry.png" style="height: 400px; margin: 20px"/></div>
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> See <a href="https://mlflow.org/docs/latest/registry.html" target="_blank">the MLflow docs</a> for more details on the model registry.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Registering a Model
# MAGIC 
# MAGIC The following workflow will work with either the UI or in pure Python.  This notebook will use pure Python.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Explore the UI throughout this lesson by clicking the "Models" tab on the left-hand side of the screen.

# COMMAND ----------

# MAGIC %md
# MAGIC Train a model and log it to MLflow. Here we are only using numeric features for simplicity of building the random forest.

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

with mlflow.start_run(run_name="LR Model") as run:
  mlflow.sklearn.log_model(lr, "model", input_example=X_train[:5]) # Add input_example for easier serving
  mlflow.log_metric("mse", mean_squared_error(y_test, lr.predict(X_test)))

# COMMAND ----------

# MAGIC %md
# MAGIC Create a unique model name so you don't clash with other workspace users.
# MAGIC 
# MAGIC Note that a registered model name must be a non-empty UTF-8 string and cannot contain forward slashes(/), periods(.), or colons(:).

# COMMAND ----------

model_name = f"{cleaned_username}_sklearn_lr"
model_name

# COMMAND ----------

# MAGIC %md
# MAGIC Register the model.

# COMMAND ----------

run_id = run.info.run_id
model_uri = f"runs:/{run_id}/model"

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC  **Open the *Models* tab on the left of the screen to explore the registered model.**  Note the following:<br><br>
# MAGIC 
# MAGIC * It logged who trained the model and what code was used
# MAGIC * It logged a history of actions taken on this model
# MAGIC * It logged this model as a first version
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/model-registry-1.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC Check the status.  It will initially be in `PENDING_REGISTRATION` status.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_version_details = client.get_model_version(name=model_name, version=1)

model_version_details.status

# COMMAND ----------

# MAGIC %md
# MAGIC Now add a model description

# COMMAND ----------

client.update_registered_model(
  name=model_details.name,
  description="This model forecasts Airbnb housing list prices based on various listing inputs."
)

# COMMAND ----------

# MAGIC %md
# MAGIC Add a version-specific description.

# COMMAND ----------

client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version was built using OLS linear regression with sklearn."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploying a Model
# MAGIC 
# MAGIC The MLflow Model Registry defines several model stages: `None`, `Staging`, `Production`, and `Archived`. Each stage has a unique meaning. For example, `Staging` is meant for model testing, while `Production` is for models that have completed the testing or review processes and have been deployed to applications. 
# MAGIC 
# MAGIC Users with appropriate permissions can transition models between stages. 

# COMMAND ----------

# MAGIC %md
# MAGIC Now that you've learned about stage transitions, transition the model to the `Production` stage.

# COMMAND ----------

import time

time.sleep(30) # In case the registration is still pending

# COMMAND ----------

client.transition_model_version_stage(
  name=model_details.name,
  version=model_details.version,
  stage="Production",
)

# COMMAND ----------

# MAGIC %md
# MAGIC Fetch the model's current status.

# COMMAND ----------

model_version_details = client.get_model_version(
  name=model_details.name,
  version=model_details.version,
)
print(f"The current model stage is: '{model_version_details.current_stage}'")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Fetch the latest model using a `pyfunc`.  Loading the model in this way allows us to use the model regardless of the package that was used to train it.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> You can load a specific version of the model too.

# COMMAND ----------

import mlflow.pyfunc

model_version_uri = f"models:/{model_name}/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC Apply the model.

# COMMAND ----------

model_version_1.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploying a New Model Version
# MAGIC 
# MAGIC The MLflow Model Registry enables you to create multiple model versions corresponding to a single registered model. By performing stage transitions, you can seamlessly integrate new model versions into your staging or production environments.

# COMMAND ----------

# MAGIC %md
# MAGIC Create a new model version and register that model when it's logged.

# COMMAND ----------

from sklearn.linear_model import Ridge

with mlflow.start_run(run_name="LR Ridge Model") as run:
  alpha = .9
  ridgeRegression = Ridge(alpha=alpha)
  ridgeRegression.fit(X_train, y_train)
  
  # Specify the `registered_model_name` parameter of the `mlflow.sklearn.log_model()`
  # function to register the model with the MLflow Model Registry. This automatically
  # creates a new model version
  
  mlflow.sklearn.log_model(
    sk_model=ridgeRegression,
    artifact_path="sklearn-ridge-model",
    registered_model_name=model_name,
  )
  mlflow.log_param("alpha", alpha)
  mlflow.log_metric("mse", mean_squared_error(y_test, ridgeRegression.predict(X_test)))

# COMMAND ----------

# MAGIC %md
# MAGIC Put the new model into the staging branch.

# COMMAND ----------

import time

time.sleep(30)

client.transition_model_version_stage(
  name=model_details.name,
  version=2,
  stage="Staging",
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Check the UI to see the new model version.
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/model-registry-2.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC Use the search functionality to grab the latest model version.

# COMMAND ----------

model_version_infos = client.search_model_versions(f"name = '{model_name}'")
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])

# COMMAND ----------

# MAGIC %md
# MAGIC Add a description to this new version.

# COMMAND ----------

client.update_model_version(
  name=model_name,
  version=new_model_version,
  description="This model version is a ridge regression model with an alpha value of .9 that was trained in scikit-learn."
)

# COMMAND ----------

# MAGIC %md
# MAGIC Since this model is now in staging, you can execute an automated CI/CD pipeline against it to test it before going into production.  Once that is completed, you can push that model into production.

# COMMAND ----------

client.transition_model_version_stage(
  name=model_name,
  version=new_model_version,
  stage="Production",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Archiving and Deleting
# MAGIC 
# MAGIC You can now archive and delete old versions of the model.

# COMMAND ----------

client.transition_model_version_stage(
  name=model_name,
  version=1,
  stage="Archived",
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Delete version 1.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> You cannot delete a model that is not first archived.

# COMMAND ----------

client.delete_model_version(
  name=model_name,
  version=1
)

# COMMAND ----------

# MAGIC %md
# MAGIC Archive version 2 of the model too.

# COMMAND ----------

client.transition_model_version_stage(
  name=model_name,
  version=2,
  stage="Archived",
)

# COMMAND ----------

# MAGIC %md
# MAGIC Now delete the entire registered model.

# COMMAND ----------

client.delete_registered_model(model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review
# MAGIC **Question:** How does MLflow tracking differ from the model registry?  
# MAGIC **Answer:** Tracking is meant for experimentation and development.  The model registry is designed to take a model from tracking and put it through staging and into production.  This is often the point that a data engineer or a machine learning engineer takes responsibility for the deployment process.
# MAGIC 
# MAGIC **Question:** Why do I need a model registry?  
# MAGIC **Answer:** Just as MLflow tracking provides end-to-end reproducibility for the machine learning training process, a model registry provides reproducibility and governance for the deployment process.  Since production systems are mission critical, components can be isolated with ACL's so only specific individuals can alter production models.  Version control and CI/CD workflow integration is also a critical dimension of deploying models into production.
# MAGIC 
# MAGIC **Question:** What can I do programmatically versus using the UI?  
# MAGIC **Answer:** Most operations can be done using the UI or in pure Python.  A model must be tracked using Python, but from that point on everything can be done either way.  For instance, a model logged using the MLflow tracking API can then be registered using the UI and can then be pushed into production.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Topics & Resources
# MAGIC 
# MAGIC **Q:** Where can I find out more information on MLflow Model Registry?  
# MAGIC **A:** Check out <a href="https://mlflow.org/docs/latest/registry.html" target="_blank">the MLflow documentation</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
