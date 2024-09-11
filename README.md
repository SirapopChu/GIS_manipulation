# GIS_manipulation
GIS Data manipulation Systems and Deep learning model for analyzing weather environment data. ground cover or other physical information To study risks and impacts on business units or organizations, integrating geographic and economic data sets.

# Requirements
requirements.txt

# Overview of Systems
A layers of RNN-model call Rnet is used for predicting impact levels in the range (1-4) using the weather data from open_meteo, which 1 is the least severe and 4 is the most severe. The disasters that the model can predict include droughts, floods, and storms in various districts. By referring to the latitude and longitude coordinate system of Thailand using the Open data Disaster Dataset by the Department of Disaster Prevention and Mitigation (DDPM) (reference to open_meteo.ipynb and RNet.ipynb) . After that, in I-net, we will use the latitude and longitude disaster level information from the original dataset to make a template risk score map using the inverse distance weight (IDW) technique, referring to the risk level. For entire space of Thailand then we matching the risk score of entire map to point of interest (POI) and sample raster values from POI lat-lon then rematch it too Financial Data too predict (reference to IDW_DDPM_loc.ipynb, lat-lon_firmscrap.ipynb, I-net)
## tools in Rnet
