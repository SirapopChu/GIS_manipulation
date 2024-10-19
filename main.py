import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import keras

import keplerfunction as kf

## System config
firm_name = 'hmpro'
period = (2018, 2023)
forecast_steps = 10

input_folder = './input'
output_file_name = f'{firm_name}_disaster_firm_location_forecast'
output_folder = f'./output/{firm_name}'

# Input data
firm_location = kf.read_data(input_folder=input_folder, dir_name='firm_location', firm_name=firm_name) # Your firm location
firm_finance = kf.read_data(input_folder=input_folder, dir_name='firm_finance', firm_name=firm_name) # Your firm financial index

disaster_occurrence_file = r'input\disaster_occurance\disaster_occurence.csv' # Disaster occurrence file
disaster_occurrence = pd.read_csv(disaster_occurrence_file)

## 2. Disaster firm index
disaster_firm_index = kf.calculate_disaster_index(firm_loc=firm_location, dis_occ=disaster_occurrence, period=period)


## 3. Predict Firm's Disaster coefficients(DQC) and Disaster importances(DAP) on Financial Indexes.
firm_finance_feature = kf.calculate_firm_finance_features(firm_finance, fin_col_range=(1, -3), period=period)

# Predict firm's DQC and DAP
firm_finance_model = kf.modeling(firm_name=firm_name, firm_finance_feature=firm_finance_feature, disaster_firm_index=disaster_firm_index)
DQC_df = firm_finance_model.find_DQC() # Disaster Quantity Coefficients (Q)
DAP_df = firm_finance_model.find_DAP() # Disaster Affection Probability (P)


## 4. Firm's Disaster Occurences Index Forecasting

# Load saved model
wd_test = keras.models.load_model(r'forecast_net\do_forecast_models\wd_forecast_model.keras')
fl_test = keras.models.load_model(r'forecast_net\do_forecast_models\fl_forecast_model.keras')
pm_test = keras.models.load_model(r'forecast_net\do_forecast_models\pm_forecast_model.keras')
dis_occ_models = [wd_test, fl_test, pm_test]

# Forecast
FDO = kf.DO_forecast(firm_loc=firm_location, dis_occ=disaster_occurrence, dis_occ_models=dis_occ_models, period=period, steps=forecast_steps)


## 5. Inference location-based

# location DO -> Forecast -> inference Q and P
firm_loc_forecast = kf.calculate_disaster_location(firm_loc=firm_location, dis_occ=disaster_occurrence,
                                                dis_occ_models=dis_occ_models, period=period, predict_step=[3, 5, 10])


output_file = os.path.join(output_folder, output_file_name)
firm_loc_forecast.to_json(f'{output_file}.json', orient='records', force_ascii=False)
print(f'firm results saved to {output_file}')