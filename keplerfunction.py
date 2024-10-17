import os
import numpy as np 
import pandas as pd
from scipy.spatial import cKDTree

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

import warnings 
warnings.filterwarnings("ignore")

def get_disaster_in_year(dis_occ, type, year) -> pd.DataFrame:
    return dis_occ[(dis_occ['year'] == year) & (dis_occ['type'] == type)]

def idw_transfer(disaster_df, business_df, value_col, neighbors=3, power=2):
    """
    Transfers values from the disaster_df to the business_df based on Inverse Distance Weighting (IDW).

    Parameters:
    disaster_df : pd.DataFrame - DataFrame containing disaster locations and the value column.
    business_df : pd.DataFrame - DataFrame containing business branch locations.
    value_col : str - The column in disaster_df to be transferred.
    neighbors : int - The number of nearest disaster locations to consider for each business branch.
    power : int - The power parameter for IDW.

    Returns:
    business_df : pd.DataFrame - The business DataFrame with an additional column for the transferred values.
    """

    # Extract coordinates
    disaster_coords = disaster_df[['latitude', 'longitude']].values
    business_coords = business_df[['lat', 'lon']].values
    
    # Build KD-tree for fast neighbor lookup
    tree = cKDTree(disaster_coords)
    
    # Find the nearest neighbors for each business location
    distances, indices = tree.query(business_coords, k=neighbors)
    
    # Initialize an array to store weighted values
    weighted_values = np.zeros(len(business_df))
    
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if neighbors == 1:  # Handle the case where only one neighbor is found
            dist = np.array([dist])
            idx = np.array([idx])
        
        # Avoid division by zero by setting very small distances to a minimum value
        dist[dist == 0] = 1e-10
        
        # Apply IDW formula
        weights = 1 / dist**power
        weights /= weights.sum()  # Normalize weights
        
        # Weighted average of the disaster value column
        weighted_values[i] = np.sum(weights * disaster_df[value_col].values[idx])
    
    # Add the new column with IDW values to the business DataFrame
    business_df[f'{value_col}_idw'] = weighted_values
    
    return business_df

def calculate_disaster_score(disaster_firm, select_col, stat='mean'):
    
    if stat == 'mean':
        return np.round(disaster_firm[select_col].mean())
    
    if stat == 'sum':
        return np.round(disaster_firm[select_col].sum())

# 2.Disaster firm index
def calculate_disaster_index(firm_loc, dis_occ, period: tuple[int]) -> pd.DataFrame:
    
    disaster_types: list[str] = dis_occ['type'].unique().tolist()
    
    disaster_firm_index: dict  = {}
    
    for disaster_type in disaster_types:
        
        disaster_firm_index[disaster_type] = {}
        
        for year in range(period[0], period[1]+1):
            
            disaster_data: pd.DataFrame = get_disaster_in_year(dis_occ=dis_occ, type=disaster_type, year=year)
            
            firm_disaster_affect: pd.DataFrame = idw_transfer(disaster_df=disaster_data, business_df=firm_loc, value_col='n_occurence', neighbors=5, power=1)
            
            firm_disaster_score: pd.DataFrame = calculate_disaster_score(firm_disaster_affect, 'n_occurence_idw', stat='sum')
            
            disaster_firm_index[disaster_type][year] = firm_disaster_score
            
    return pd.DataFrame(disaster_firm_index)

# 3.Find Disaster Quantity Coefficients (Q)
def calculate_firm_finance_features(firm_fin: pd.DataFrame, fin_col_range: tuple[int, int], period: tuple[int, int],  fin_scale_factor: int = 1_000_0000) -> pd.DataFrame:
    
    period_year: list[int] = np.arange(period[0], period[1]+1)
    
    firm_fin_filter_year: pd.DataFrame = firm_fin[firm_fin['Year'].isin(period_year)]
    
    firm_fin_filter_feature: pd.DataFrame = firm_fin_filter_year.iloc[:, fin_col_range[0]:fin_col_range[1]+1]
    
    return firm_fin_filter_feature / fin_scale_factor
    
class modeling:
    """
    คลาสสำหรับการสร้างและประเมินโมเดลต่างๆ
    """
    def __init__(self, firm_finance_feature: pd.DataFrame, disaster_firm_index: pd.DataFrame):
        self.firm_finance_feature = firm_finance_feature
        self.disaster_firm_index = disaster_firm_index
        self.finance_features = firm_finance_feature.columns

    def generate_X_Y(self, prices: np.ndarray, disaster_occurrences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        สร้างชุดข้อมูล X และ Y สำหรับการฝึกโมเดล
        """
        features = []  # เพื่อเก็บ tuples ของฟีเจอร์
        targets = []   # เพื่อเก็บความแตกต่างของราคา

        # ลูปผ่านแถวของข้อมูล โดยเริ่มจาก index 1
        for i in range(1, len(prices)):
            features.append(disaster_occurrences[i])  # X เป็นฟีเจอร์ที่ index ปัจจุบัน
            targets.append(prices[i] - prices[i - 1])  # Y เป็นความแตกต่างของราคา

        return np.asarray(features), np.asarray(targets)

    def make_y_logistic(self, y_data: np.ndarray) -> np.ndarray:
        """
        แปลง Y data เป็นค่าลอจิสติก (1 ถ้า y_i > 0, มิฉะนั้น 0)
        """
        return np.asarray([1 if y_i > 0 else 0 for y_i in y_data])

    def find_DQC(self, output_path: str = 'output/homepro_disaster_quantity_coefficients.csv') -> pd.DataFrame:
        """
        ค้นหาค่าสัมประสิทธิ์ปริมาณภัยพิบัติ (Disaster Quantity Coefficients - DQC) โดยใช้ Linear Regression
        """
        output_folder = output_path.split('/')[0]
        DQC_result = []

        for feature_idx in range(self.firm_finance_feature.shape[1]):
            feature_result = []
            feature_name = self.finance_features[feature_idx]

            # สร้าง X และ Y
            X, y = self.generate_X_Y(
                prices=self.firm_finance_feature.iloc[:, feature_idx].to_numpy(),
                disaster_occurrences=self.disaster_firm_index.to_numpy()
            )

            # ตรวจสอบว่ามีข้อมูลเพียงพอสำหรับการฝึกโมเดลหรือไม่
            if len(X) == 0:
                print(f"Insufficient data for feature {feature_name}. Skipping...")
                continue

            # สร้างและฝึกโมเดล Linear Regression
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            # คำนวณค่าประเมินโมเดล
            rmse = np.sqrt(root_mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            coefs = model.coef_
            bias = model.intercept_

            # เก็บผลลัพธ์
            feature_result.append(feature_name)
            feature_result.append(r2)
            feature_result.append(rmse)
            feature_result.extend(coefs)
            feature_result.append(bias)

            DQC_result.append(feature_result)

        # สร้าง DataFrame จากผลลัพธ์
        DQC_columns = ['finance_feature', 'r2', 'rmse'] + [f'coef_{col}' for col in self.disaster_firm_index.columns] + ['bias']
        DQC_df = pd.DataFrame(DQC_result, columns=DQC_columns)

        # บันทึกผลลัพธ์ลงไฟล์ CSV

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        DQC_df.to_csv(output_path, index=False)
        print(f"DQC results saved to {output_path}")
        return DQC_df

    def find_DAP(self, output_path: str = 'output/homepro_disaster_affect_probabilities.csv') -> pd.DataFrame:
        """
        ค้นหาความน่าจะเป็นของการได้รับผลกระทบจากภัยพิบัติ (Disaster Affection Probability - DAP) โดยใช้ Random Forest Regressor
        """
        output_folder = output_path.split('/')[0]
        DAP_result = []

        for feature_idx in range(self.firm_finance_feature.shape[1]):
            feature_result = []
            feature_name = self.finance_features[feature_idx]

            # สร้าง X และ Y
            X, y = self.generate_X_Y(
                prices=self.firm_finance_feature.iloc[:, feature_idx].to_numpy(),
                disaster_occurrences=self.disaster_firm_index.to_numpy()
            )

            # ตรวจสอบว่ามีข้อมูลเพียงพอสำหรับการฝึกโมเดลหรือไม่
            if len(X) == 0:
                print(f"Insufficient data for feature {feature_name}. Skipping...")
                continue

            # สร้างและฝึกโมเดล Random Forest Regressor
            model = RandomForestRegressor()
            model.fit(X, y)
            y_pred = model.predict(X)

            # คำนวณค่าประเมินโมเดล
            rmse = np.sqrt(root_mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            fi = model.feature_importances_

            # เก็บผลลัพธ์
            feature_result.append(feature_name)
            feature_result.append(r2)
            feature_result.append(rmse)
            feature_result.extend(fi)

            DAP_result.append(feature_result)

        # สร้าง DataFrame จากผลลัพธ์
        DAP_columns = ['finance_feature', 'r2', 'rmse'] + [f'fi_{col}' for col in self.disaster_firm_index.columns]
        DAP_df = pd.DataFrame(DAP_result, columns=DAP_columns)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # บันทึกผลลัพธ์ลงไฟล์ CSV
        DAP_df.to_csv(output_path, index=False)
        print(f"DAP results saved to {output_path}")
        return DAP_df

# 4. Forecasting Disaster Occurences Index
def forecast_occurrence(model, base_list: np.ndarray, steps: int):
    
    predictions = []

    for _ in range(steps):
        pred = model.predict(base_list.reshape(1, -1, 1), verbose=0)
        pred = np.round(pred)
        predictions.append(pred)
        base_list = np.append(base_list[1:], pred)

    predictions = np.array(predictions).reshape(-1, 1)

    return predictions

def DO_forecast(firm_loc, dis_occ, dis_occ_models: list, period: tuple[int], steps: int = 10 ) -> pd.DataFrame:
    
    disaster_types: list[str] = dis_occ['type'].unique().tolist()
    
    disaster_forecast: dict  = {}
    
    for disaster_type, dis_occ_model in zip(disaster_types, dis_occ_models):
        
        disaster_forecast[disaster_type] = {}

        disaster_array = []
        
        for year in range(period[0], period[1]+1):
            
            disaster_data: pd.DataFrame = get_disaster_in_year(dis_occ=dis_occ, type=disaster_type, year=year)

            firm_disaster_affect: pd.DataFrame = idw_transfer(disaster_df=disaster_data, business_df=firm_loc, value_col='n_occurence', neighbors=5, power=1)
            
            firm_disaster_score: pd.DataFrame = calculate_disaster_score(firm_disaster_affect, 'n_occurence_idw', stat='mean')
                        
            disaster_array.append(firm_disaster_score)
        
        disaster_array = np.array(disaster_array)
        
        forecasted_array = forecast_occurrence(model=dis_occ_model, base_list=disaster_array, steps=steps)
        
        disaster_forecast[disaster_type] = np.append(disaster_array, forecasted_array.reshape(-1))
        
    disaster_forecast_df = pd.DataFrame(disaster_forecast)
    
    year = np.arange(period[0], period[1]+1+steps).tolist()
    
    disaster_forecast_df['year'] = year
    
    disaster_forecast_df['tag'] = ['reference']*len(np.arange(period[0], period[1]+1))+ ['forecast']*steps
    
    return disaster_forecast_df

# 5. Inference location-based
# Disaster - Future Firm Q and P
def inference_disocc_to_QP(fdo: pd.DataFrame, dqc: pd.DataFrame) -> pd.DataFrame:
    
    for i, finance_feature in enumerate(dqc['finance_feature']):
        
        feature_i_result = []
        
        for j in range(len(fdo)):
        
            gain_loss = dqc['coef_wind'][i]*fdo['wind'][j] \
                            + dqc['coef_flood'][i]*fdo['flood'][j] \
                            + dqc['coef_pm'][i]*fdo['pm'][j] \
                            + dqc['bias'][i]
        
            feature_i_result.append(gain_loss)
        
        fdo[finance_feature] = feature_i_result
        
    return fdo

# location DO -> Forecast -> inference Q and P 
def calculate_disaster_location(firm_loc, dis_occ, dis_occ_models: list, period: tuple[int], predict_step= list[int]) -> pd.DataFrame:
    
    disaster_types: list[str] = dis_occ['type'].unique().tolist()
    
    for disaster_type in disaster_types:
        
        dis_occ[disaster_type] = {}
    
        for year in range(period[0], period[1]+1):
            
            disaster_data: pd.DataFrame = get_disaster_in_year(dis_occ=dis_occ, type=disaster_type, year=year)
            
            firm_disaster_affect: pd.DataFrame = idw_transfer(disaster_df=disaster_data, business_df=firm_loc, value_col='n_occurence', neighbors=5, power=1)
            
            firm_loc[f'{disaster_type}_{year}'] = firm_disaster_affect['n_occurence_idw']
            
    # Forecast
    
    year_period = [year for year in range(period[0], period[1]+1)]
    
    forecast_list = []
    
    for disaster_type, dis_occ_model in zip(disaster_types, dis_occ_models):
        
        sel_col = [f'{disaster_type}_{year}' for year in year_period]
        
        forecast_disaster = []
        
        for i in range(len(firm_loc)):
            
            data_array = firm_loc.loc[i, sel_col].values
            data_array = np.asarray(data_array).astype('float32')
            
            forecast_disaster.append(forecast_occurrence(dis_occ_model, data_array, steps=predict_step[-1]))
            
        forecast_list.append(forecast_disaster)
            
    forecast_array = np.asarray(forecast_list)
    
    for disaster_index, disaster_type in enumerate(disaster_types):
        
        for year_step in predict_step:
            
            firm_loc[f'{disaster_type}_{year_period[-1]+year_step}'] = forecast_array[disaster_index, :, year_step-1]
            
    return firm_loc
