import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import keras

from keplerfunction import calculate_disaster_index, calculate_firm_finance_features, modeling, DO_forecast, inference_disocc_to_QP, calculate_disaster_location

## System config
period = (2018, 2023)
forecast_steps = 10
output_file_name = 'homepro_disaster_firm_location_forecast'
output_folder = './output'

# Import dataset
firm_location_file = r'analysis\dataset\firm_location\location_homepro.csv' # Your firm location
firm_finance_file = r'analysis\dataset\firm_finance\HMPRO - SET - 2018-2023.csv' # Your firm financial index
disaster_occurrence_file = r'analysis\dataset\disaster_occurance\disaster_occurence.csv' # Disaster occurrence file

firm_location = pd.read_csv(firm_location_file)
firm_finance = pd.read_csv(firm_finance_file)
disaster_occurrence = pd.read_csv(disaster_occurrence_file)


## 2. Disaster firm index
disaster_firm_index = calculate_disaster_index(firm_loc=firm_location, dis_occ=disaster_occurrence, period=period)


## 3. Predict Firm's Disaster coefficients(DQC) and Disaster importances(DAP) on Financial Indexes.
firm_finance_feature = calculate_firm_finance_features(firm_finance, fin_col_range=(1, -3), period=period)

# Predict firm's DQC and DAP
firm_finance_model = modeling(firm_finance_feature=firm_finance_feature, disaster_firm_index=disaster_firm_index)
DQC_df = firm_finance_model.find_DQC() # Disaster Quantity Coefficients (Q)
DAP_df = firm_finance_model.find_DAP() # Disaster Affection Probability (P)


## 4. Firm's Disaster Occurences Index Forecasting

# Load saved model
wd_test = keras.models.load_model(r'forecast_net\do_forecast_models\wd_forecast_model.keras')
fl_test = keras.models.load_model(r'forecast_net\do_forecast_models\fl_forecast_model.keras')
pm_test = keras.models.load_model(r'forecast_net\do_forecast_models\pm_forecast_model.keras')
dis_occ_models = [wd_test, fl_test, pm_test]

# Forecast
FDO = DO_forecast(firm_loc=firm_location, dis_occ=disaster_occurrence, dis_occ_models=dis_occ_models, period=period, steps=forecast_steps)


## 5. Inference location-based

# location DO -> Forecast -> inference Q and P
firm_loc_forecast = calculate_disaster_location(firm_loc=firm_location, dis_occ=disaster_occurrence,
                                                dis_occ_models=dis_occ_models, period=period, predict_step=[3, 5, 10])


output_file = os.path.join(output_folder, output_file_name)
firm_loc_forecast.to_json(f'{output_file}.json', orient='records', force_ascii=False)
print(f'firm results saved to {output_file}')