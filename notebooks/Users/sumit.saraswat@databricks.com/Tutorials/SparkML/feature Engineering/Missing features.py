# Databricks notebook source
# MAGIC %md
# MAGIC # Types of missing data

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 1.  Missing completely at random (MCAR) - probablity of missing is same for all observations

# COMMAND ----------

import pandas as pd

# COMMAND ----------

df = pd.read_csv('/dbfs/FileStore/user/sumit.saraswat@databricks.com/train.csv')

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Find missing patterns in attributes

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Missing Data Not at random (MNAR) - There is a relationship that exist between the attributes missing values

# COMMAND ----------

# MAGIC %md
# MAGIC #### Finding the percentage of data sets where values are null for Cabin attribute 

# COMMAND ----------

import numpy as np
df['cabin_null']=np.where(df['Cabin'].isnull(),1,0)
print('percentage of missing values in attribute Cabin are' ,df['cabin_null'].mean()*100)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating a delta table out of titanic dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table titanic_bronze

# COMMAND ----------

# MAGIC 
# MAGIC %sql
# MAGIC 
# MAGIC create table if not exists tempTitanic
# MAGIC   USING csv
# MAGIC   OPTIONS (path "/FileStore/user/sumit.saraswat@databricks.com/train.csv", header "true", inferSchema "true");
# MAGIC 
# MAGIC create table if not exists titanic_bronze using delta
# MAGIC    tblproperties (delta.autoOptimize.optimizeWrite = true, delta.autoOptimize.autoCompact = true) 
# MAGIC    as select * from tempTitanic;
# MAGIC      
# MAGIC select * from titanic_bronze;

# COMMAND ----------

# MAGIC %md
# MAGIC lets check if the cabin missing and a person surviving has any relationship. Assumption is since this dataset is gathered after the accident we will find cabin values missing values when the person does not survive

# COMMAND ----------

# MAGIC %sql
# MAGIC select survived, mean(case when isnull(Cabin)=true then 1 else 0 end)*100 as cabin_null_percentage  from titanic_bronze group by survived;

# COMMAND ----------

# MAGIC %md 
# MAGIC if we do the similar excercise with age , we can see age is not dependent on whether the person survived or not

# COMMAND ----------

# MAGIC %sql
# MAGIC select survived, mean(case when isnull(Age)=true then 1 else 0 end)*100 as age_null_percentage  from titanic_bronze group by survived;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Missing at Random (MAR)

# COMMAND ----------

# MAGIC %md
# MAGIC # Techniques of handling missing values

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Mean/Median/Mode Replacement : can be used for MCAR (like Age)
# MAGIC 2. Random sample Imputation
# MAGIC 3. Capturing NaN values with a new feature 
# MAGIC 4. End of Distribution Imputation
# MAGIC 5. Arbitrary imputation
# MAGIC 6. Frequent categories imputation

# COMMAND ----------

# MAGIC %sql
# MAGIC describe table titanic_bronze

# COMMAND ----------

# MAGIC %sql
# MAGIC -- refill median value of age wherever Age has a missing value
# MAGIC create table if not exists titanic_silver using delta
# MAGIC    tblproperties (delta.autoOptimize.optimizeWrite = true, delta.autoOptimize.autoCompact = true) 
# MAGIC    as select * , case when isnull(Age) then 28 else Age end  as age_calc from titanic_bronze;
# MAGIC    
# MAGIC  

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- Alter table drop Age column
# MAGIC  ALTER TABLE titanic_silver SET TBLPROPERTIES (
# MAGIC    'delta.columnMapping.mode' = 'name',
# MAGIC    'delta.minReaderVersion' = '2',
# MAGIC    'delta.minWriterVersion' = '5');
# MAGIC alter table titanic_silver drop column Age;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from titanic_silver ;

# COMMAND ----------

# MAGIC %sql 
# MAGIC select percentile(age,.5) as median_age, mean(age) from titanic_bronze ;