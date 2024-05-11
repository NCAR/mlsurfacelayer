import pandas as pd
from glob import glob
from os.path import join
from .derived import *
from pvlib.solarposition import get_solarposition
import datetime


# Marthas Vineyard Coastal Observatory data ingest and feature calculator
#DateTime
#wave_height:0_m:m
#wave_period:0_m:deg
#wave_dir:0_m:deg
#water_temp:0_m:C
#bottom_current:0_m:cm/s
#bottom_current_dir:0_m:deg_toward
#near_surf_current:0_m:cm/s
#near_surf_current_dir:0_m:deg
#press-air_pres-nominal_depth:0_m:m   (pressure minus air pressure minus nominal depth -- tide paros
#water_temp_2:0_m:C
#salinity:0_m:PSU
#num_records_per_period:12_m:count
#air_temp:12_m:C
#RH:12_m:%
#press:12_m:mb
#air_temp_median:12_m:C
#RH_median:12_m:%
#press_median:12_m:mb
#air_temp_std:12_m:C
#RH_std:12_m:%
#press_std:12_m:mb
#air_temp_len:12_m:count
#RH_len:12_m:count
#pres_len:12_m:count
#num_records_per_period:18.4_m:count
#wspd_3D1:18.4_m:m/s ( 3D1 is a sensor)
#wdir_3D1:18.4_m:deg
#w:18.4_m:m/s
#air_temp_speed_of_sound_3D1:18.4_m:C
#wspd_3D1_inst:18.4_m:m/s
#wspd_3D1_inst_2:18.4_m:m/s
#air_temp_speed_of_sound_3D1_2:18.4_m:C
#sig_U:18.4_m:UNK ( sig means 
#sig_V:18.4_m:UNK
#sig_W:18.4_m:UNK
#sig_T:18.4_m:UNK
#uv_3D1:18.4_m:m2/s2  covariance
#uw_3D1:18.4_m:m2/s2  covariance
#vw_3D1:18.4_m:m2/s2  covariance
#wT_3D1:18.4_m:m/s K  covariance vertical wind speed with air temp
#wspd_3D1_length:18.4_m:count recs between 0 100 < 4*std
#tson_3D1_length:18.4_m:count recs between -25 and 35 and < 4*std


def process_mvco_data(csv_path, out_file, nan_column="", mvco_lon=-70.544, mvco_lat=41.3435,
                        elevation=0.0, average_period=None):
    """
    This function loads all of the MVCO data and then calculates the relevant derived quantities necessary
    to build the machine learning model for parameterization of surface layer.

    Columns in derived data show follow the following convention "name_of_variable:level:units". Colons separate
    different components, and underscores separate words in each subsection. `df.columns.str.split(":").str[0]` extracts
    the variable names, `df.columns.str.split(":").str[1]` extracts levels, and `df.columns.str.split(":").str[2]`
    extracts units.

    Args:
        csv_path: Path to all csv files.
        out_file: Where derived data are written to.
        nan_column: Column used to filter bad examples.
        mvco_lat: Latitude of tower site in degrees.
        mvco_lon: Longitude of tower site in degrees.
        elevation: Elevation of site in meters.
        average_period: Window obs are averaged over.
    Returns:
        `pandas.DataFrame` containing derived data.
    """
    #
    # Read raw data
    #
    print("Reading raw data")
    csv_file = csv_path
    print (csv_file)
    raw_data = pd.read_csv(csv_file, na_values=[-9999.0])

    #
    # Create a time series index using the time
    #
    raw_data.index = pd.to_datetime(raw_data["DateTime"], format="%Y-%m-%d %H:%M:%S")

    #
    # Filter out data based on "bad" data in nan_columns
    #
    #raw_data = raw_data.loc[~pd.isna(raw_data[nan_column])]
   
    #
    # List data columns included in training dataset
    #
    derived_columns = ["zenith:0_m:degrees",
                       "azimuth:0_m:degrees",
                       "temperature:12_m:K",
                       "temperature_med:12_m:K",
                       "temperature_std:12_m:K",
                       "temperature:18.4_m:K",
                       "temperature2:18.4_m:K",
                       "water_sfc_temperature:0_m:K",
                       "pressure:12_m:hPa",
                       "pressure_med:12_m:hPa",
                       "pressure_std:12_m:hPa",
                       "potential_temperature:12_m:K",
                       "skin_virtual_potential_temperature:0_m:K",
                       "mixing_ratio:0_m:g_kg-1",
                       "mixing_ratio:12_m:g_kg-1",
                       "relative_humidity:12_m:%",
                       "wave_direction:0_m:degrees",
                       "wave_height:0_m:m",
                       "wave_period:0_m:s",
                       "wave_phase_speed:0_m:m_s-1",
                       "near_surf_current:0_m:m_s-1",
                       "near_surf_current_dir:0_m:deg",
                       "wind_speed:12_m:m_s-1",
                       "wind_speed:18.4_m:m_s-1",
                       "wind_direction:18.4_m:degrees",
                       "wind_speed_inst:18.4_m:m_s-1",
                       "wind_direction_inst:18.4_m:degrees",
                       "u_wind:18.4_m:m_s-1",
                       "v_wind:18.4_m:m_s-1",
                       "u_wave:0_m:m_s-1",
                       "v_wave:0_m:m_s-1",
                       "angle_between_wind_wave:0_m:degrees",
                       "bulk_richardson:12_m:none",
                       "surface_roughness_charnock:0_m:m",
                       "surface_roughness_drennan:0_m:m",
                       #"potential_temperature_gradient:20_m:K_m-1",
                       #"wind_speed_gradient:20_m:s-1",
                       #"wind_speed_gradient:40_m:s-1",
                       "u_w:18.4_m:m2_s-2",
                       "v_w:18.4_m:m2_s-2",
                       "friction_velocity:18.4_m:m_s-1",
                       "kinematic_sensible_heat_flux:18.4_m:K_m_s-1",
                       "temperature_scale:18.4_m:K"
                       ]

    print( "Calculating derived variables")

    #
    # Define the derived_data dataframe
    #
    derived_data = pd.DataFrame(index=raw_data.index, columns=derived_columns, dtype=float)

    #
    # Fill in solar angles
    #
    solar_data = get_solarposition(raw_data.index, mvco_lat, mvco_lon, altitude=elevation, method="nrel_numba")
    derived_data["zenith:0_m:degrees"] = solar_data["zenith"]
    derived_data["azimuth:0_m:degrees"] = solar_data["azimuth"]

    #
    # Water surface temperature
    #
    derived_data["water_sfc_temperature:0_m:K"] = celsius_to_kelvin(raw_data["water_temp:0_m:C"])

    #
    # Wave direction , height, period
    #
    derived_data["wave_direction:0_m:degrees"] = raw_data["wave_dir:0_m:deg"]
    derived_data["wave_height:0_m:m"] = raw_data["wave_height:0_m:m"]
    derived_data["wave_period:0_m:s"] = raw_data["wave_period:0_m:deg"] # note the error in the unit -- will fix 
    derived_data["wave_phase_speed:0_m:m_s-1"] = derived_data["wave_period:0_m:s"]* 9.8/(2*np.pi)

    #
    # Current variables
    #
    derived_data["near_surf_current:0_m:m_s-1"] =   raw_data["near_surf_current:0_m:cm/s"]/100
    derived_data["near_surf_current_dir:0_m:deg"] = raw_data["near_surf_current_dir:0_m:deg"]
    derived_data["bottom_current:0_m:m_s-1"] = raw_data["bottom_current:0_m:cm/s"]/100
    derived_data["bottom_current_dir:0_m:deg"] = raw_data["bottom_current_dir:0_m:deg_toward"]

    #
    # Wind Speed 
    #
    derived_data["wind_speed:18.4_m:m_s-1"] = raw_data["wspd_3D1:18.4_m:m/s"]

    #
    # Wind Direction
    #
    derived_data["wind_direction:18.4_m:degrees"] = raw_data["wdir_3D1:18.4_m:deg"]

    #
    # Derived data wind components
    #
    derived_data["u_wind:18.4_m:m_s-1"], derived_data["v_wind:18.4_m:m_s-1"] = wind_components(derived_data["wind_speed:18.4_m:m_s-1"], derived_data["wind_direction:18.4_m:degrees"])

    derived_data["u_wave:0_m:m_s-1"], derived_data["v_wave:0_m:m_s-1"] = wind_components(derived_data["wave_phase_speed:0_m:m_s-1"], derived_data["wave_direction:0_m:degrees"])

    derived_data["angle_between_wind_wave:0_m:degrees"] = 180/np.pi * np.arccos((derived_data["u_wave:0_m:m_s-1"] * derived_data["u_wind:18.4_m:m_s-1"] + derived_data["v_wave:0_m:m_s-1"] * derived_data["v_wind:18.4_m:m_s-1"])/(derived_data["wave_phase_speed:0_m:m_s-1"] * derived_data["wind_speed:18.4_m:m_s-1"]))

    #
    # Pressure
    # 
    derived_data["pressure:12_m:hPa"] = raw_data["press:12_m:mb"]
    derived_data["pressure_median:12_m:hPa"] = raw_data["press_median:12_m:mb"]
    derived_data["pressure_std:12_m:hPa"] = raw_data["press_std:12_m:mb"]

    #    
    # Temperature  
    #
    derived_data["temperature:12_m:K"] =  celsius_to_kelvin(raw_data["air_temp:12_m:C"])
    derived_data["temperature_median:12_m:K"] =  celsius_to_kelvin(raw_data["air_temp_median:12_m:C"])
    derived_data["temperature_std:12_m:K"] =  raw_data["air_temp_std:12_m:C"]
    derived_data["temperature:18.4_m:K"] =  celsius_to_kelvin(raw_data["air_temp_speed_of_sound_3D1:18.4_m:C"])
    derived_data["temperature2:18.4_m:K"] =  celsius_to_kelvin(raw_data["air_temp_speed_of_sound_3D1_2:18.4_m:C"])

    #
    # flux components
    #
    derived_data["u_w:18.4_m:m2_s-2"] = raw_data["uw_3D1:18.4_m:m2/s2"]
    derived_data["v_w:18.4_m:m2_s-2"] = raw_data["vw_3D1:18.4_m:m2/s2"]
    derived_data["w_T:18.4_m:m2_s-2"] = raw_data["wT_3D1:18.4_m:m/s K"]


    #    
    # Relative humidity 
    #
    derived_data["relative_humidity:12_m:%"]=  raw_data['RH:12_m:%']
    derived_data["relative_humidity_median:12_m:%"]=  raw_data['RH_median:12_m:%']
    derived_data["relative_humidity_std:12_m:%"]=  raw_data['RH_std:12_m:%']
   
    #
    # Sea Surface/ Skin  mixing ratio (RH = 100%)  1.293 = air density
    # Note: 12 [m] * 9.81 [m/s^2] * 1.293 [kg/m^3] = 152.21196 Pa 
    sea_surface_pressure = derived_data["pressure:12_m:hPa"] + 152.21196/100
    derived_data["mixing_ratio:0_m:g_kg-1"] = mixing_ratio(raw_data["water_temp:0_m:C"], 100,  sea_surface_pressure)

    #
    # Virtual potential skin temperature : use sea surface temp
    #
    derived_data[ "skin_virtual_potential_temperature:0_m:K"] = virtual_temperature( derived_data["water_sfc_temperature:0_m:K"], derived_data["mixing_ratio:0_m:g_kg-1"])


    #
    # Derive potential temperature     #
    derived_data["potential_temperature:12_m:K"] = potential_temperature(derived_data["temperature:12_m:K"], derived_data[f"pressure:12_m:hPa"])

    #
    # Mixing ratio
    #
    derived_data["mixing_ratio:12_m:g_kg-1"] = mixing_ratio( derived_data["temperature:12_m:K"]-273, derived_data["relative_humidity:12_m:%"], derived_data[f"pressure:12_m:hPa"])

    #
    # Friction Velocity: 40 (given in raw data), 60 and 80m (derived from u*=(〈u'w'〉^2+〈v'w'〉^2)^1/4
    #
    derived_data["u_w:18.4_m:m2_s-2"] = raw_data["uw_3D1:18.4_m:m2/s2"]

    derived_data["v_w:18.4_m:m2_s-2"] = raw_data["vw_3D1:18.4_m:m2/s2"]

    derived_data["friction_velocity:18.4_m:m_s-1"]= ((raw_data['uw_3D1:18.4_m:m2/s2'])**2 +  (raw_data['vw_3D1:18.4_m:m2/s2'])**2 )**(.25)

    derived_data["kinematic_sensible_heat_flux:18.4_m:K_m_s-1"] = raw_data["wT_3D1:18.4_m:m/s K"]

    derived_data["temperature_scale:18.4_m:K"] = derived_data["kinematic_sensible_heat_flux:18.4_m:K_m_s-1"]/derived_data["friction_velocity:18.4_m:m_s-1"]

    #
    # define surface roughness as a functio of friction velocity, wave height , and wave phase speed
    # http://waveworkshop.org/13thWaves/Papers/COWCLIP_paper.pdf
    #search "Drennan et al. (2003)"
    #z0 = 3.35 * derived_data["wave_height:0_m:m"] * (derived_data["friction_velocity:18.4_m:m_s-1"]/derived_data["wave_phase_speed:0_m:m_s-1"] )**3.4
    derived_data["surface_roughness_drennan:0_m:m"] = 3.35 * derived_data["wave_height:0_m:m"] * (derived_data["friction_velocity:18.4_m:m_s-1"]/derived_data["wave_phase_speed:0_m:m_s-1"] )**3.4

    # Charnock's relation
    #z0 =  αc u*2/g 
    derived_data["surface_roughness_charnock:0_m:m"] = .015/9.8 * derived_data["friction_velocity:18.4_m:m_s-1"]**2

    #
    # d = Zero-plane displacement is the height in meters above the ground at which zero mean wind speed 
    # is achieved as a result of flow obstacles such as trees or buildings.
    #
    #d = derived_data["wave_height:0_m:m"] 
    d = 0
    #
    # https://en.wikipedia.org/wiki/Log_wind_profile
    # 
    z0 = derived_data["surface_roughness_drennan:0_m:m"]
    derived_data[ "wind_speed:12_m:m_s-1"] = derived_data[ "wind_speed:18.4_m:m_s-1"] * np.log((12 - d )/z0)/np.log((18.4 - d)/z0);

    #
    # Bulk Richardson's number Note that wspd is at a diff height 
    #  
    derived_data[ "bulk_richardson:12_m:none"] = bulk_richardson_number( derived_data["potential_temperature:12_m:K"], 12,
                                                                         derived_data["mixing_ratio:12_m:g_kg-1"],
                                                                         derived_data["skin_virtual_potential_temperature:0_m:K"],
                                                                         derived_data["wind_speed:12_m:m_s-1"])

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


