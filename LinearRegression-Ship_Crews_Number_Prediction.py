#!/usr/bin/env python
# coding: utf-8

# In[9]:


#Import the findspark module and initialize it with the specified Spark path
import findspark
findspark.init('/home/mina/python-spark/spark-3.4.0-bin-hadoop3/')

#Import the pyspark module and the SparkSession class
import pyspark
from pyspark.sql import SparkSession

#Create a Spark session with the specified app name
spark = SparkSession.builder.appName('shipCrews').getOrCreate()


# In[10]:


#Read a CSV file named 'cruise_ship_info.csv' into a DataFrame
#The 'inferSchema=True' option infers data types for columns, and 'header=True' treats the first row as column names
dataset = spark.read.csv('cruise_ship_info.csv', inferSchema=True, header=True)

#Print the schema of the dataset
dataset.printSchema()


# In[11]:

# Retrieve and display the first 3 rows of the dataset
dataset.head(3)


# In[12]:


#Iterate over the first 3 rows of the dataset and print each row to underestand that cruise lines may have an effect on how many crew memebers we need
for record in dataset.head(3):
    print(record , '\n')


# In[13]:

#Group the DataSet by the 'Cruise_line' column, count cruise line we have in dataset
#Some of Cruise_line is more important than others
dataset.groupBy('Cruise_line').count().show()


# In[14]:


# Import the necessary module for string indexing
from pyspark.ml.feature import StringIndexer

# Create a StringIndexer transformer to index the 'Cruise_line' column
indexer = StringIndexer(inputCol="Cruise_line" , outputCol="Cruise_line_Indexed")

# Fit the indexer model to the dataset to generate an indexing model
indexerModel = indexer.fit(dataset)

# Transform the dataset using the indexing model to add the indexed column
index_df = indexerModel.transform(dataset)


# Display the first 3 rows of the DataFrame after string indexing
index_df.head(3)


# In[15]:

#Print the schema of the DataFrame after string indexing
index_df.printSchema()


# In[17]:

#Retrieve and display the column names of the DataFrame after string indexing
index_df.columns


# In[18]:

#Import the necessary modules for creating feature vectors and vector assembly
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

#Create a VectorAssembler to assemble selected columns into a feature vector
assembler = VectorAssembler(inputCols=['Cruise_line_Indexed',
 'Age',
 'Tonnage',
 'passengers',
 'length',
 'cabins',
 'passenger_density'] , outputCol='ShipFeatures')

# Transform the DataFrame using the VectorAssembler to add the 'ShipFeatures' column
output = assembler.transform(index_df)

# Display the first row of the transformed DataFrame
output.head(1)


# In[19]:

# Select the 'shipFeatures' and 'crew' columns from the transformed DataFrame
final_data = output.select('shipFeatures', 'crew')
final_data.show()


# In[20]:

# Split the final_data DataFrame into training and testing datasets
# using a split ratio of 70% for training data and 30% for testing data
train_data, test_data = final_data.randomSplit([0.7,0.3])


# Display summary statistics of the 'train_data' DataFrame
train_data.describe().show()


# In[21]:

# Display summary statistics of the 'test_data' DataFrame
test_data.describe().show()


# In[22]:

# Import the necessary module for linear regression
from pyspark.ml.regression import LinearRegression

# Create a LinearRegression model with specified features, label, and prediction columns
lr = LinearRegression(featuresCol='shipFeatures', labelCol='crew', predictionCol='CrewNumber')

# Fit the LinearRegression model to the training data
lrModel = lr.fit(train_data)

# Evaluate the model on the test data and store the results
test_result = lrModel.evaluate(test_data)

# Display the residuals (differences between predicted and actual values) of the model on the test data
test_result.residuals.show()


# In[23]:

# Access and display the root mean squared error (RMSE) from the test_result
test_result.rootMeanSquaredError


# In[24]:

# Access and display the R-squared (coefficient of determination) from the test_result
test_result.r2


# In[25]:

# Display summary statistics of the 'final_data' DataFrame
final_data.describe().show()


# In[27]:


from pyspark.sql.functions import corr
# Calculate and display the correlation between 'crew' and 'passengers' columns from main dataset
dataset.select(corr('crew' , 'passengers')).show()


# In[28]:

# Calculate and display the correlation between 'crew' and 'cabins' columns from main dataset
dataset.select(corr('crew' , 'cabins')).show()

