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

#Decision tree model desription 





