import pandas as pd
from glob import glob
from os.path import join
from .derived import *
from pvlib.solarposition import get_solarposition
import datetime


#Time
#temperature:12.54_m:C
#wind_speed:13.49_m:m/s
#wind_direction:13.49_m:deg
#water_surface_temperature:-3.5_m:C
#pressure:0_m:mb
#relative_humidity:12.54_m:%
#solar_downwelling_flux:?:W/m2
#IR_downwelling_flux:?:W/m2
#COARE_sensible_heat_flux:?:W/m2
#COARE_latent_heat_flux:?:W/m2
#COARE_Obukhov_length_scale:?:m
#COARE_Ustar:?:m/s
#wu_NOAA_sonic:13.49_m:m/s
#wv_NOAA_sonic:13.49_m:m/s
#wt_NOAA_sonic:13.49_m:m/s
#wq_NOAA_sonic:13.49_m:m/s
#wu_NDS1_sonic:11.54_m:m/s
#wv_NDS1_sonic:11.54_m:m/s
#wt_NDS1_sonic:11.54_m:m/s
#wu_NDS0_sonic:9.69_m:m/s
#wv_NDS0_sonic:9.69_m:m/s
#wt_NDS0_sonic:9.69_m:m/s
#sensible_heat_covariance_NOAA_sonic:13.49_m:W/m2
#sensible_heat_ID_NOAA_sonic:13.49_m:W/m2
#latent_heat_covariance_NOAA_sonic:13.49_m:W/m2
#latent_heat_ID_NOAA_sonic:13.49_m:W/m2
#sensible_heat_covariance_NDS1_sonic:11.54_m:W/m2
#sensible_heat_ID_NDS1_sonic:11.54_m:W/m2
#sensible_heat_covariance_NDS0_sonic:9.69_m:W/m2
#sensible_heat_ID_NDS0_sonic:9.69_m:W/m2
#Latitude
#Longitude
#azimuth:0_m:degrees
#zenith:0_m:degrees


# Research Vessel Sally Ride Data

# lat lon of 36.700N 122.343W buoy 46114 off coast of Monteray, CA
# Fairly central to the data gathering area of the CASPER West project
def process_rvsr2_data(csv_path, out_file, nan_column="", svsr_lon=-122.343, svsr_lat=36.700,
                        elevation=0.0, average_period=None):
    """
    This function loads all of the RVSR data and then calculates the relevant derived quantities necessary
    to build the machine learning model for parameterization of surface layer.

    Columns in derived data show follow the following convention "name_of_variable:level:units". Colons separate
    different components, and underscores separate words in each subsection. `df.columns.str.split(":").str[0]` extracts
    the variable names, `df.columns.str.split(":").str[1]` extracts levels, and `df.columns.str.split(":").str[2]`
    extracts units.
    # Sample data line: 2017-09-28 16:31:06.099994,,,,,16.55852,,,,,,,,,,1013.39734,,-0.025269268,0.008753834,-0.06390887,,,,,,,,,,

    Args:
        csv_path: Path to all csv files.
        out_file: Where derived data are written to.
        nan_column: Column used to filter bad examples.
        svsr_lat: Latitude of tower site in degrees. This is approximate for the data gathering interval
        svsr_lon: Longitude of tower site in degrees. This is approximate for the data gathering interval
        elevation: Elevation of site in meters. Seal level
        average_period: Window obs are averaged over. 20 minute data. No further averaging
    Returns:
        `pandas.DataFrame` containing derived data.
    """
    #
    # Read raw data
    #
    print("Reading raw data")
    csv_file = csv_path
    print (csv_file)
    raw_data = pd.read_csv(csv_file)

    #
    # Create a time series index using the time
    #
    raw_data.index = pd.to_datetime(raw_data["Time"], format="%Y-%m-%d %H:%M:%S")

    #
    # Filter out data based on "bad" data in nan_columns
    #
    #raw_data = raw_data.loc[~pd.isna(raw_data[nan_column])]
   
    #
    # List data columns included in training dataset
    #
    derived_columns = ["latitude:0_m:degrees",
                       "longitude:0_m:degrees",
                       "zenith:0_m:degrees",
                       "azimuth:0_m:degrees",
                       "temperature:12.54_m:K",
                       "water_sfc_temperature:-3.5_m:K",
                       "pressure:0_m:hPa",
                       "pressure:12.54_m:hPa",
                       "potential_temperature:12.54_m:K",
                       "skin_virtual_potential_temperature:0_m:K",
                       "mixing_ratio:0_m:g_kg-1",
                       "mixing_ratio:12.54_m:g_kg-1",
                       "relative_humidity:12.54_m:%",
                       "wind_speed:13.49_m:m_s-1",
                       "wind_direction:13.49_m:degrees",
                       "u_wind:13.49_m:m_s-1",
                       "v_wind:13.49_m:m_s-1",
                       "bulk_richardson:12_m:none",
                       "solar_downwelling_flux:?:W_m-2",
                       "IR_downwelling_flux:?:W_m-2",
                       "COARE_sensible_heat_flux:?:W_m-2",
                       "COARE_latent_heat_flux:?:W_m-2",
                       "COARE_Obukhov_length_scale:?:m",
                       "COARE_Ustar:?:m_s-1",
                       "wu_NOAA_sonic:13.49_m:m_s-1",
                       "wv_NOAA_sonic:13.49_m:m_s-1",
                       "wt_NOAA_sonic:13.49_m:m_s-1",
                       "wq_NOAA_sonic:13.49_m:m_s-1",
                       "wu_NDS1_sonic:11.54_m:m_s-1",
                       "wv_NDS1_sonic:11.54_m:m_s-1",
                       "wt_NDS1_sonic:11.54_m:m_s-1",
                       "wu_NDS0_sonic:9.69_m:m_s-1",
                       "wv_NDS0_sonic:9.69_m:m_s-1",
                       "wt_NDS0_sonic:9.69_m:m_s-1",
                       "ustar_NOAA_sonic:13.49_m:m_s-1",
                       "ustar_NDS1_sonic:11.54_m:m_s-1",
                       "ustar_NDS0_sonic:9.69_m:m_s-1",
                       "sensible_heat_covariance_NOAA_sonic:13.49_m:W_m-2",
                       "sensible_heat_ID_NOAA_sonic:13.49_m:W_m-2",
                       "latent_heat_covariance_NOAA_sonic:13.49_m:W_m-2",
                       "latent_heat_ID_NOAA_sonic:13.49_m:W_m-2",
                       "sensible_heat_covariance_NDS1_sonic:11.54_m:W_m-2"
                       "sensible_heat_ID_NDS1_sonic:11.54_m:W_m-2",
                       "sensible_heat_covariance_NDS0_sonic:9.69_m:W_m-2",
                       "sensible_heat_ID_NDS0_sonic:9.69_m:W_m-2"
                       ]

    print( "Calculating derived variables")

    #
    # Define the derived_data dataframe
    #
    derived_data = pd.DataFrame(index=raw_data.index, columns=derived_columns, dtype=float)

    #
    # Latitude and longitude
    #
    derived_data["latitude:0_m:degrees"] = raw_data["Latitude"]
    derived_data["longitude:0_m:degrees"] = raw_data["Longitude"]

    #
    # Fill in solar angles
    #
    derived_data["zenith:0_m:degrees"] = raw_data["zenith:0_m:degrees"]
    derived_data["azimuth:0_m:degrees"] = raw_data["azimuth:0_m:degrees"]

    #
    # Water surface temperature
    #
    #sea_surface_temperature:0_m:C
    derived_data["water_sfc_temperature:-3.5_m:K"] = celsius_to_kelvin(raw_data["water_surface_temperature:-3.5_m:C"])

    #
    # Wind Speed 
    #
    derived_data["wind_speed:13.49_m:m_s-1"] = raw_data["wind_speed:13.49_m:m/s"]

    #
    # Wind Direction
    #
    derived_data["wind_direction:13.49_m:degrees"] = raw_data["wind_direction:13.49_m:deg"]

    #
    # Derived data wind components
    #
    derived_data["u_wind:13.49_m:m_s-1"], derived_data["v_wind:13.49_m:m_s-1"] = wind_components(derived_data["wind_speed:13.49_m:m_s-1"], derived_data["wind_direction:13.49_m:degrees"])

    #
    # Pressure
    # 
    derived_data["pressure:0_m:hPa"] = raw_data["pressure:0_m:mb"]
    # Note: 12.54 [m] * 9.81 [m/s^2] * 1.293 [kg/m^3] =   159.06
    derived_data["pressure:12.54_m:hPa"] =  derived_data["pressure:0_m:hPa"] - 159.06

    #    
    # Temperature  
    #
    derived_data["temperature:12.54_m:K"] = celsius_to_kelvin(raw_data["temperature:12.54_m:C"])

    #    
    # Relative humidity 
    #
    derived_data["relative_humidity:12.54_m:%"]=  raw_data["relative_humidity:12.54_m:%"]
   
    #
    # Sea surface mixing ratio 
    #
    derived_data["mixing_ratio:0_m:g_kg-1"] = mixing_ratio(raw_data["water_surface_temperature:-3.5_m:C"], 100, derived_data["pressure:12.54_m:hPa"])

    #
    # Virtual potential skin temperature 
    #
    derived_data[ "skin_virtual_potential_temperature:0_m:K"] = virtual_temperature( derived_data["water_sfc_temperature:-3.5_m:K"], derived_data["mixing_ratio:0_m:g_kg-1"])


    #
    # potential temp
    #
    derived_data["potential_temperature:12.54_m:K"] = potential_temperature(derived_data["temperature:12.54_m:K"], derived_data[f"pressure:12.54_m:hPa"])

    #
    # Mixing ratio
    #
    derived_data["mixing_ratio:12.54_m:g_kg-1"] = mixing_ratio( raw_data["temperature:12.54_m:C"], derived_data["relative_humidity:12.54_m:%"], derived_data[f"pressure:12.54_m:hPa"])

    #
    # Bulk Richardson's number Note that wspd is at a diff height 
    #  
    derived_data[ "bulk_richardson:12.54_m:none"] = bulk_richardson_number( derived_data["potential_temperature:12.54_m:K"], 12,
                                                                         derived_data["mixing_ratio:12.54_m:g_kg-1"],
                                                                         derived_data["skin_virtual_potential_temperature:0_m:K"],
                                                                         derived_data["wind_speed:13.49_m:m_s-1"])

    #
    # define surface roughness as a functio of friction velocity, wave height , and wave phase speed
    # http://waveworkshop.org/13thWaves/Papers/COWCLIP_paper.pdf
    #search "Drennan et al. (2003)"
    #z0 = 3.35 * derived_data["wave_height:0_m:m"] * (derived_data["friction_velocity:18.4_m:m_s-1"]/derived_data["wave_phase_speed:0_m:m_s-1"] )**3.4
    #derived_data["surface_roughness_drennan:0_m:m"] = 3.35 * derived_data["wave_height:0_m:m"] * (derived_data["friction_velocity:18.4_m:m_s-1"]/derived_data["wave_phase_speed:0_m:m_s-1"] )**3.4

    # Charnock's relation
    #z0 =  αc u*2/g 
    #derived_data["surface_roughness_charnock:0_m:m"] = .015/9.8 * derived_data["friction_velocity:18.4_m:m_s-1"]**2

    #
    # d = Zero-plane displacement is the height in meters above the ground at which zero mean wind speed 
    # is achieved as a result of flow obstacles such as trees or buildings.
    #
    #d = derived_data["wave_height:0_m:m"] 
    #d = 0
    #
    # https://en.wikipedia.org/wiki/Log_wind_profile
    # 
    #z0 = derived_data["surface_roughness_drennan:0_m:m"]
    #derived_data[ "wind_speed:12_m:m_s-1"] = derived_data[ "wind_speed:18.4_m:m_s-1"] * nKKp.log((12 - d )/z0)/np.log((18.4 - d)/z0);


    derived_data["solar_downwelling_flux:?:W_m-2"] = raw_data["solar_downwelling_flux:?:W/m2"]
    derived_data["IR_downwelling_flux:?:W_m-2"] = raw_data["IR_downwelling_flux:?:W/m2"]
    derived_data["COARE_sensible_heat_flux:?:W_m-2"] = raw_data["COARE_sensible_heat_flux:?:W/m2"]
    derived_data["COARE_latent_heat_flux:?:W_m-2"] = raw_data["COARE_latent_heat_flux:?:W/m2"]
    derived_data["COARE_Obukhov_length_scale:?:m"] = raw_data["COARE_Obukhov_length_scale:?:m"]
    derived_data["COARE_Ustar:?:m_s-1"] = raw_data["COARE_Ustar:?:m/s"]
    derived_data["wu_NOAA_sonic:13.49_m:m_s-1"] = raw_data["wu_NOAA_sonic:13.49_m:m/s"]
    derived_data["wv_NOAA_sonic:13.49_m:m_s-1"] = raw_data["wv_NOAA_sonic:13.49_m:m/s"]
    derived_data["wt_NOAA_sonic:13.49_m:m_s-1"] = raw_data["wt_NOAA_sonic:13.49_m:m/s"]
    derived_data["wq_NOAA_sonic:13.49_m:m_s-1"] = raw_data["wq_NOAA_sonic:13.49_m:m/s"]
    derived_data["wu_NDS1_sonic:11.54_m:m_s-1"] = raw_data["wu_NDS1_sonic:11.54_m:m/s"]
    derived_data["wv_NDS1_sonic:11.54_m:m_s-1"] = raw_data["wv_NDS1_sonic:11.54_m:m/s"]
    derived_data["wt_NDS1_sonic:11.54_m:m_s-1"] = raw_data["wt_NDS1_sonic:11.54_m:m/s"]
    derived_data["wu_NDS0_sonic:9.69_m:m_s-1"] = raw_data["wu_NDS0_sonic:9.69_m:m/s"]
    derived_data["wv_NDS0_sonic:9.69_m:m_s-1"] = raw_data["wv_NDS0_sonic:9.69_m:m/s"] 
    derived_data["wt_NDS0_sonic:9.69_m:m_s-1"] = raw_data["wt_NDS0_sonic:9.69_m:m/s"]
    derived_data["sensible_heat_covariance_NOAA_sonic:13.49_m:W_m-2"] = raw_data["sensible_heat_covariance_NOAA_sonic:13.49_m:W/m2"]
    derived_data["sensible_heat_ID_NOAA_sonic:13.49_m:W_m-2"] = raw_data["sensible_heat_ID_NOAA_sonic:13.49_m:W/m2"]
    derived_data["latent_heat_covariance_NOAA_sonic:13.49_m:W_m-2"] = raw_data["latent_heat_covariance_NOAA_sonic:13.49_m:W/m2"]
    derived_data["latent_heat_ID_NOAA_sonic:13.49_m:W_m-2"] = raw_data["latent_heat_ID_NOAA_sonic:13.49_m:W/m2"]
    derived_data["sensible_heat_covariance_NDS1_sonic:11.54_m:W_m-2"] = raw_data["sensible_heat_covariance_NDS1_sonic:11.54_m:W/m2"]
    derived_data["sensible_heat_ID_NDS1_sonic:11.54_m:W_m-2"] = raw_data["sensible_heat_ID_NDS1_sonic:11.54_m:W/m2"] 
    derived_data["sensible_heat_covariance_NDS0_sonic:9.69_m:W_m-2"] = raw_data["sensible_heat_covariance_NDS0_sonic:9.69_m:W/m2"]
    derived_data["sensible_heat_ID_NDS0_sonic:9.69_m:W_m-2"] = raw_data["sensible_heat_ID_NDS0_sonic:9.69_m:W/m2"]

    #
    # Friction Velocity:  u*=(〈u'w'〉^2+〈v'w'〉^2)^1/4
    #
    derived_data["ustar_NOAA_sonic:13.49_m:m_s-1"] =(derived_data["wu_NOAA_sonic:13.49_m:m_s-1"]**2 + derived_data["wv_NOAA_sonic:13.49_m:m_s-1"]**2)**.25
    derived_data["ustar_NDS1_sonic:11.54_m:m_s-1"] =(derived_data["wu_NDS1_sonic:11.54_m:m_s-1"]**2 + derived_data["wv_NDS1_sonic:11.54_m:m_s-1"]**2)**.25
    derived_data["ustar_NDS0_sonic:9.69_m:m_s-1"] = (derived_data["wu_NDS0_sonic:9.69_m:m_s-1"]**2 + derived_data["wv_NDS0_sonic:9.69_m:m_s-1"]**2)**.25


    #
    # Create rolling average of data columns if requested
    #
    if average_period is not None:
        derived_data = derived_data.rolling(window=average_period).mean()
        #derived_data = derived_data.dropna()

    #
    # Output data
    #
    #derived_data = derived_data.dropna()
    derived_data.to_csv(out_file, columns=derived_columns, index_label="Time")

    return derived_data
def load_derived_data(filename,
                      train_test_split_date, dropna=True, filter_counter_gradient=False):
    """
    Load derived data file, remove NaN events, and split the data into training and test sets.

    Args:
        filename: Name of the derived data csv file.
        train_test_split_date: Date where data are split into training and testing sets
        dropna: Whether to drop NaN fields or not.
        filter_counter_gradient: Remove datapoints with counter gradient fluxes

    Returns:
        dict: data divided into input, output, and derived with training and testing sets
    """
    all_data = pd.read_csv(filename, index_col="Time", parse_dates=["Time"])
    if dropna:
        all_data = all_data.dropna()
    if filter_counter_gradient:
        all_data = filter_counter_gradient_data(all_data)
    data = dict()
    data["train"] = all_data.loc[all_data.index < pd.Timestamp(train_test_split_date)]
    data["test"] = all_data.loc[all_data.index >= pd.Timestamp(train_test_split_date)]
    return data

def load_derived_data_random_test_train(filename, dropna=False, filter_counter_gradient=False):
    """
    Load derived data file, remove NaN events, and split the data into training and test sets.

    Args:
        filename: Name of the derived data csv file.
        train_test_start_date: begin date for data 
        train_test_end_date: end date for data 
        dropna: Whether to drop NaN fields or not.
        filter_counter_gradient: Remove datapoints with counter gradient fluxes

    Returns:
        dict: data divided into input, output, and derived with training and testing sets
    """
    all_data = pd.read_csv(filename, index_col="Time", parse_dates=["Time"])
    all_data =  all_data[~all_data.index.duplicated(keep='first')]
    
    all_data = all_data.dropna(subset=[ "zenith:0_m:degrees", "azimuth:0_m:degrees", "temperature:12_m:K", "water_sfc_temperature:0_m:K", "pressure:12_m:hPa", "potential_temperature:12_m:K", "skin_virtual_potential_temperature:0_m:K", "mixing_ratio:0_m:g_kg-1", "mixing_ratio:12_m:g_kg-1", "relative_humidity:12_m:%", "wave_direction:0_m:degrees", "wave_height:0_m:m", "wave_period:0_m:s", "wave_phase_speed:0_m:m_s-1", "wind_speed:18.4_m:m_s-1", "wind_direction:18.4_m:degrees", "angle_between_wind_wave:0_m:degrees","bulk_richardson:12_m:none"])
    
    #if dropna:
    #     all_data = all_data.dropna()    
    #if filter_counter_gradient:
    #    all_data = filter_counter_gradient_data(all_data)
    data = dict()

    #
    # For repeatability we use the same set of evenly spaced test weeks  
    #
    testWeeks=[1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]
    
    data["test"] = all_data.loc[all_data.index.isocalendar().week.isin(testWeeks)]
    data["train"] = all_data.loc[all_data.index.difference(data["test"].index) ]


    train = pd.DataFrame()
    train = all_data.loc[all_data.index.isocalendar().week.isin(testWeeks)]
    train.to_csv("/Volumes/SuesRoo/mvco_mlsl/mvco_train.csv", na_rep = '?')
    test = pd.DataFrame()
    test = all_data.loc[all_data.index.difference(data["test"].index) ]
    test.to_csv("/Volumes/SuesRoo/mvco_mlsl/mvco_test.csv", na_rep = '?')

    print("data loaded")
    return data
def filter_counter_gradient_data(data, gradient_column="potential_temperature_gradient:20_m:K_m-1",
                                 flux_column="sensible_heat_flux:40_m:W_m-2"):
    """
    Only keep data points where the sign of the temperature gradient matches the sign of the heat flux.

    Args:
        data:
        gradient_column:
        flux_column:

    Returns:

    """
    filtered_indices = data[gradient_column] * data[flux_column] >= 0
    filtered_data = data.loc[filtered_indices, :]
    return filtered_data


