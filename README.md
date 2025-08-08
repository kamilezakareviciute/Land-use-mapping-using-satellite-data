# Land-use-mapping-using-satellite-data
This project analyses the feasability of Copernicus Sentinel-2 satellite data use for landuse mapping using NDVI index and machine learning methods in Jonava and Kėdainiai. The analysed period is vegetation period from March to November for years 2020-2021. 

First, satellite data is retrieved from Google Earth Engine using JavaSript. Data collections are downloaded to local machine for further procesing and point data extraction in ArcGis. Point data in csv format is used to create machine learning model in Python. Sample data is used to train desicion tree method and classify landuse data. Maps are created in ArcGis Pro to analyse spatial distribution of each class. Real landuse data is used for validation of the results. 

Sentinel-2 satellite data is retrieved from Google Earth Engine software using JavaSript https://code.earthengine.google.com/ 

#Area of interest and selection of Sentinel-2 data
var area = 
ee.Geometry.Polygon(
        [[[24.204882261943553, 55.10540665561003],
          [24.204882261943553, 55.03090014311963],
          [24.360064146709178, 55.03090014311963],
          [24.360064146709178, 55.10540665561003]]], null, false),
    s2 = ee.ImageCollection("COPERNICUS/S2_SR")

#Below code is used to filter cloudy data and select satellite bands to produce NDVI value for each pixel
var collection = s2
  .filterBounds(area)
  .filterDate('2020-03-01', '2021-11-01')
  .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", 30)
  .select('B4','B8')   

  var addNDVI = function(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
}
#Creating NDVI immage collection from calclulated bands 
var ndvi = collection.map(addNDVI).select('NDVI');

#Calclulated NDVI images are saved to local Google drive. Each task is run manually to download raster images

var batch = require('users/fitoprincipe/geetools:batch'); 
ee.ImageCollection = ndvi
batch.Download.ImageCollection.toDrive(ndvi,'Folder',
{type:'float'}); 

Raster images are inspected for errors and cloud impacts. If inspection passed .TIFF format images are worked in ArcGis Pro to extract NDVI data from raster to point. Raster to Multipoint funtion was applied to create one csv file, containing NDVI values for each point covering 10m grid. Columns contain date of observation and rows pointID. Data set was seperated for machine learning model training and data used for classification. Training sample is extracted from NDVI contains known landuse classes: deciduous trees, evergreen trees, grass, urbanised fabric and water bodies. 

#Data for decision tree model is located in zip "NDVI data for model". Location of the files should be modified. 

import pandas as pd
import os
from scipy.stats import randint
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn._config import get_config, set_config
import numpy as np

#A total of 3 decision tree models are created for further analysis:
#First - features NDVI from year 2021 and is used to classify landuse types in one town - Jonava only;
#Second - features NDVI dataset that has been run trough Savitzky-Golay filter, that iliminates NDVI noise trough use of averages. Used to classify landuse types on one town Jonava in 2021;
#Third - uses monthly NDVI values of 2021 Jonava. Is used to classify landuse types in two towns - Jonava and Kėdainiai in years 2021 and 2022. 

#FIRST MODEL

#Importing data sets for decision tree training and final classification. Adjust directories accordingly
dataset_for_model_learning =pd.read_csv(r"\NDVI_for_model_training_JONAVA_2021.csv")
data_for_classification =pd.read_csv(r"\NDVI_for_model_classification_JONAVA_2021.csv") 


NDVI_values_for_training = dataset_for_model_learning.iloc[:,5:-1] #selecting only NDVI observations for training sample
class_for_training = dataset_for_model_learning['klase'] #training sample by class feature
NDVI_values_for_classification = data_for_classification.iloc[:,2:-2] #dataset is used for classification

#training data set is split into datasets for training and testing. Testing sample size is 30% 
NDVI_value_train, NDVI_value_test, class_value_train, class_value_test = train_test_split(NDVI_values_for_training, class_for_training, test_size=0.3)

dt = DecisionTreeClassifier()
dt = dt.fit(NDVI_value_train, class_value_train) #training DT
class_predictions = dt.predict(NDVI_value_test) #testing trained DT against test datasets
acc_DT = accuracy_score(class_value_test, class_predictions) #Getting accuracy score from the predictions 

#search of best parameters - tailor DT model
param_rs = {"max_depth": randint(1,5),
            "max_features": randint(1, 23),
            "min_samples_leaf": randint(1, 200),
            "criterion": ["gini", "entropy"],
            "min_samples_split": randint(200, 500),
            "ccp_alpha" : [0.005]
         }
#looking for best parameters specified in list
RS = RandomizedSearchCV(dt, param_rs, cv=5, n_iter=50)


#optimised new DT model with new best parameters
optimised_DT = RS.fit(NDVI_value_train, class_value_train)
class_predictions_optimised = optimised_DT.predict(NDVI_value_test)
acc_score_optimised_DT = accuracy_score(class_value_test, class_predictions_optimised)
best_params = RS.best_params_ #getting best parameters for optimised DT

#Classifying NDVI data set with optimised DT model 
classification_optimised = optimised_DT.predict(NDVI_values_for_classification)
classification_optimised_list = classification_optimised.tolist()

#creating new csv file with predicted values 
column_value = pd.Series(classification_optimised_list)
data_for_classification.insert(loc=0, column='klase', value=column_value)
data_for_classification.to_csv(r'\NDVI_for_model_classification_JONAVA_2021_classified.csv', index = None, header=True)

#best parameter export to csv and visualisation of the tree
scores = optimised_DT.cv_results_
scores_dframe = pd.DataFrame(scores).sort_values('rank_test_score')
scores_dframe.to_csv(r'X:\ArcGis Pro\Bakaluras_galutinis\bakalauras_1\Jonava_2021\Be_filtro\For GIT\NDVI for model\Optimised_DT_scores.csv', index = None, header=True) #adjust exoprt location

#extracting only dates without extra information and class used to classify 
class_names = ['vanduo', 'pieva', 'lapuociai','spygliuociai','infrastruktura']
feature_names = ['2021_03_05', '2021_03_27', '2021_04_19','2021_04_21','2021_05_01','2021_05_11','2021_05_31'
                ,'2021_06_10','2021_06_18','2021_06_20','2021_06_23','2021_06_30','2021_07_10'
                ,'2021_07_15','2021_07_20','2021_07_25','2021_09_08','2021_09_11','2021_09_26'
                ,'2021_10_06','2021_10_08','2021_10_11','2021_10_18'  
                ]

#visualising optimised decision tree with pyplot
dt_visuals = RS.best_estimator_
fig = plt.figure(figsize=(20,10)) 
tree.plot_tree(dt_visuals, class_names=class_names,feature_names =feature_names, filled=True,fontsize=10)
plt.show()

#graphviz is required for graph export. Graphviz install https://graphviz.org/download/
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin' #location of installed graphviz

export_graphviz(dt_visuals, out_file='Classification_visualisation.dot',
                feature_names=feature_names,
                class_names=class_names,
                filled=True, rounded=True, special_characters=True)
!dot -Tpng Classification_visualisation.dot -o Classification_visualisation.png

#SECOND MODEL 

#Exact same principle as the fist one, only datasets for training and classification are different 

#THIRD MODEL 

