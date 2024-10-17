import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, BatchNormalization, Dropout, MaxPooling1D, Flatten, Reshape

np.random.seed(42)

# Config
feature_folder = './dataset/open-meteo' # dataset docs https://open-meteo.com/en/docs
target_file = 'dataset/ddpm_amphoe_hazard_dataset_int.csv'
start_date = '2011-01-01'
end_date = '2020-12-31'
hazard_df = pd.read_csv(target_file)

#Functions
def import_all_files(folder_name: str, start_date: str) -> dict[str, pd.DataFrame]:
    """
    Import all files from your folder to a dictionary using your file's name as
    the key and file's content(csv) as a value for easier time-series analysis.

    Args:
    folder (str): Your to be used folder.
    start_date (str): A starting date of time series.

    Returns:
    feature_dict (dictionary): [feature_name: keys, feature(df): values]
    """
    feature_files = os.listdir(folder_name)
    feature_names = []
    feature_paths =[]

    for file in feature_files:
        feature_path = os.path.join(folder_name, file)
        feature_paths.append(feature_path)

        feature_name = file.split(start_date)[0][:-1]
        feature_names.append(feature_name)

    feature_dict = dict.fromkeys(feature_names)
    for ind, name in enumerate(feature_dict):
        df = pd.read_csv(feature_paths[ind], index_col=['latitude', 'longitude'])
        df.columns = pd.to_datetime(df.columns)
        feature_dict[name] = df

    return feature_dict

def get_data_from_dict(feature_dict: dict) -> list[pd.DataFrame]:
    """
    Get open-meteo data from dictionary that created from 'import_all_files' function, collected in a list.

    Args:
    feature_dict (dict): dictionary that has been created from 'import_all_files' function.

    Returns:
    feature_df_list (list): A list of feature dataframes extracted from 'feature_dict'.
    """

    # apparent_temperature_max_df = feature_dict['apparent_temperature_max']
    # apparent_temperature_mean_df = feature_dict['apparent_temperature_mean']
    # apparent_temperature_min_df = feature_dict['apparent_temperature_min']
    # daylight_duration_df = feature_dict['daylight_duration']
    # precipitation_hours_df = feature_dict['precipitation_hours']
    # snowfall_sum_df = feature_dict['snowfall_sum']
    # sunrise_df = feature_dict['sunrise']
    # sunset_df = feature_dict['sunset']
    # sunshine_duration_df = feature_dict['sunshine_duration']
    # weather_code_df = feature_dict['weather_code']
    # wind_direction_10m_dominant_df = feature_dict['wind_direction_10m_dominant']
    et0_fao_evapotranspiration_df = feature_dict['et0_fao_evapotranspiration']
    precipitation_sum_df = feature_dict['precipitation_sum']
    rain_sum_df = feature_dict['rain_sum']
    shortwave_radiation_sum_df = feature_dict['shortwave_radiation_sum']
    temperature_2m_max_df = feature_dict['temperature_2m_max']
    temperature_2m_mean_df = feature_dict['temperature_2m_mean']    
    temperature_2m_min_df = feature_dict['temperature_2m_min']
    wind_gusts_10m_max_df = feature_dict['wind_gusts_10m_max']
    wind_speed_10m_max_df = feature_dict['wind_speed_10m_max']

    feature_df_list = [
    et0_fao_evapotranspiration_df.copy(),
    precipitation_sum_df.copy(),
    rain_sum_df.copy(),
    shortwave_radiation_sum_df.copy(),
    temperature_2m_max_df.copy(),
    temperature_2m_mean_df.copy(),
    temperature_2m_min_df.copy(),
    wind_gusts_10m_max_df.copy(),
    wind_speed_10m_max_df.copy()
    ]

    return feature_df_list

def get_X_scaled_list() -> list[pd.DataFrame]:
    """
    Get feature data by calling 'get_data_from_dict' and scales all features using standardization method and collected in a list.

    Args: None

    Returns:
    X_scaled_list (list): A list of scaled feature dataframes.
    """ 
    # Scale features for training.
    feature_dict = import_all_files(folder_name=feature_folder, start_date=start_date)
    feature_df_list = get_data_from_dict(feature_dict=feature_dict)
    # Filter out the empty table/df and Tranpose the df to the right shape
    feature_df_list = [feature for feature in feature_df_list if feature.all().mean() != 0]
    feature_df_list = [feature.T for feature in feature_df_list]

    # Seperate category and numerical features
    categories = []
    categories_mean = [feature.mean().mean() for feature in categories]

    numerical = [feature for feature in feature_df_list if feature.mean().mean() not in categories_mean]

    # Standardization the numerical features
    numerical_scaled = [(feature.mean().mean() - feature) / feature.std().std() for feature in numerical]

    # Fuse all features together
    X_scaled_list = numerical_scaled + categories

    return X_scaled_list

# Deep Learning
# P'want suggestion 3 (Classification)

class RNet:
    """
    Deep learning LSTM model that classify hazard risk.
    """
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):

        model = Sequential()

        model.add(tf.keras.Input(shape=(self.input_shape)))

        model.add(Conv1D(filters=32, kernel_size=1, data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2, strides=1, padding="same"))
        model.add(Dropout(0.2))

        model.add(Conv1D(filters=64, kernel_size=1, data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2, strides=1, padding="same"))
        model.add(Dropout(0.2))

        model.add(LSTM(64, return_sequences=True))

        model.add(Conv1D(filters=9, kernel_size=1))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2, strides=1, padding="same"))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Reshape((3, 5)))
        model.add(Dense(self.num_classes, activation='softmax'))

        opt = keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        return model
    
    def train(self, x_train, y_train, epochs=100, batch_size=128, validation_data=None):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)
    
    # Example usage:
    # timesteps = X_train.shape[1]
    # features = X_train.shape[2]
    # model = RNet(input_shape=(timesteps, features), num_classes=5)
    # model.train(x_train, y_train, epochs=100)
    # model.evaluate(x_test, y_test)

# X and y data is on the different table, so we'll split the train and test of the main table first
# to determine which location would be on train and test dataframe then we'll split them to X_train, X_test, y_train and y_test.

train_df, test_df = train_test_split(hazard_df, test_size=0.3, random_state=42, stratify=hazard_df['pcode'])

X_scaled_list = get_X_scaled_list()

X_train_list = [feature.T for feature in [feature.T.iloc[train_df.index] for feature in X_scaled_list]]
X_train = np.array(X_train_list).T
X_train = np.asarray(X_train).astype('float32')

X_test_list = [feature.T for feature in [feature.T.iloc[test_df.index] for feature in X_scaled_list]]
X_test = np.array(X_test_list).T
X_test = np.asarray(X_test).astype('float32')

y = hazard_df[['drought_risk_level', 'flood_risk_level', 'windstorm_risk_level']]

# ordinal encoder
ord_enc = OrdinalEncoder()
y_ord = ord_enc.fit_transform(y)

num_classes = 5

y_train_list = y_ord[train_df.index]
y_train = np.array(y_train_list)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

y_test_list = y_ord[test_df.index]
y_test = np.array(y_test_list)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train_drought shape: {y_train.shape}')
print(f'y_test_drought shape: {y_test.shape}')