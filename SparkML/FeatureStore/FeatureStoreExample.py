# Databricks notebook source
filePath = "dbfs:/FileStore/sumit.saraswat/train.csv"

df_train = spark.read.csv(filePath, header="true", inferSchema="true", multiLine="true", escape='"')

display(df_train)


# COMMAND ----------

from pyspark.sql.functions import regexp_extract, udf,col
# Extract Title from Name, store in column "Title"
df = df_train.withColumn('Title',regexp_extract(col('Name'),'([A-Za-z]+)\.',1))

# Sanitise and group titles
df = df.replace(['Mlle','Mme', 'Ms', 'Dr','Master','Major','Lady','Dona','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                ['Miss','Miss','Miss','Mr','Mr', 'Mr', 'Mrs',  'Mrs', 'Mrs', 'Other',  'Other','Other','Mr','Mr','Mr'])

# COMMAND ----------


df = df.withColumn('Has_Cabin', df.Cabin.isNotNull())

# COMMAND ----------


df = df.withColumn("Family_Size", col('SibSp') + col('Parch') + 1)

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE DATABASE IF NOT EXISTS feature_store_titanic;

# COMMAND ----------

from databricks.feature_store import feature_table
from databricks.feature_store import FeatureStoreClient
fs = FeatureStoreClient()

fs.create_table(
    name="feature_store_titanic.titanic_passengers_features_2",
    primary_keys = ["Name","Cabin"],
    df = df,
    description = "Titanic Passenger Features")

# COMMAND ----------

fs.write_table(
  name = "feature_store_titanic.titanic_passengers_features_2",
  df = df,
  mode = "merge")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a training dataset

# COMMAND ----------

from databricks.feature_store import FeatureLookup

titanic_features_table = "feature_store_titanic.titanic_passengers_features_2"

# We choose to only use 2 of the newly created features
titanic_features_lookups = [
    FeatureLookup( 
      table_name = titanic_features_table,
      feature_names = "Title",
      lookup_key = ["Name","Cabin"],
    ),
    FeatureLookup( 
      table_name = titanic_features_table,
      feature_names = "Has_Cabin",
      lookup_key = ["Name","Cabin"],
    ),
#     FeatureLookup( 
#       table_name = titanic_features_table,
#       feature_names = "Family_Size",
#       lookup_key = ["Name"],
#     ),
]

# Create the training set that includes the raw input data merged with corresponding features from both feature tables
exclude_columns = ["Name", "PassengerId","Parch","SibSp","Ticket"]
training_set = fs.create_training_set(
                df_train,
                feature_lookups = titanic_features_lookups,
                label = 'Survived',
                exclude_columns = exclude_columns
                )

# COMMAND ----------

# MAGIC %md
# MAGIC ###Use the LightBGM classifier train the mode

# COMMAND ----------

from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
import lightgbm as lgb
import mlflow
from sklearn.metrics import accuracy_score

import pandas as pd

# Load the TrainingSet into a dataframe which can be passed into sklearn for training a model
training_df = training_set.load_df()

# End any existing runs (in the case this notebook is being run for a second time)
mlflow.end_run()

# Start an mlflow run, which is needed for the feature store to log the model
mlflow.start_run(run_name="lgbm_feature_store") 

data = training_df.toPandas()
data_dum = pd.get_dummies(data, drop_first=True)

# Extract features & labels
X = data_dum.drop(["Survived"], axis=1)
y = data_dum.Survived

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

lgb_params = {
            'n_estimators': 50,
            'learning_rate': 1e-3,
            'subsample': 0.27670395476135673,
            'colsample_bytree': 0.6,
            'reg_lambda': 1e-1,
            'num_leaves': 50, 
            'max_depth': 8, 
            }

mlflow.log_param("hyper-parameters", lgb_params)
lgbm_clf  = lgb.LGBMClassifier(**lgb_params)
lgbm_clf.fit(X_train,y_train)
lgb_pred = lgbm_clf.predict(X_test)

accuracy=accuracy_score(lgb_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, lgb_pred)))
mlflow.log_metric('accuracy', accuracy)
