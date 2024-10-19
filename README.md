# GIS_manipulation
  The GIS Data Manipulation Systems and Deep learning models are used to analyze weather environment data, ground cover, or other physical information to study risks and impacts on business units or organizations, by integrating geographic and economic data sets.

# Requirements
requirements.txt

# Overview of Systems
  Layers of a deep learning model called RNet are used for predicting impact levels from range 1 to 4 using an open-source weather API from open_meteo, in which 1 is the least severe and 4 is the most severe. The disasters that the model can predict include droughts, floods, and storms in various districts. Using Thailand's latitude and longitude coordinate data from the open-source, Disaster Dataset from the Department of Disaster Prevention and Mitigation (DDPM). (a reference to open_meteo.ipynb and RNet.ipynb). After RNet, INet uses the latitude and longitude risk level information from the original dataset to make a template risk score map using the inverse distance weight (IDW) technique for the risk score to cover the entirety of Thailand. Then match the point of interest (POI) to the location on the template to use the risk score on the exact location. Then merge POI lat-lon with their Financial Data to predict the business's 12 financial indexes in the latest year. (a reference to IDW_DDPM_loc.ipynb, lat-lon_firmscrap.ipynb)

## meaning
1.archive : keep an old testing  

2.do_data-extraction : developing extract API Disaster Occurrance (DO) from DDPM 

3.forecast_net : development directory , Disaster Occurance forecast 

4.main : main of execute process flow line 

5.keplerfunction : every class in main function 

6.output : directory for collect output from main 
