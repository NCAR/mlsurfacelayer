import pandas as pd
from glob import glob
from os.path import join
from .derived import *
from pvlib.solarposition import get_solarposition
import datetime



#Time,
#WindSpeed:1.4_m:m/s,
#WindSpeed:2.74_m:m/s,
#WindDir:1.4_m:deg,
#WindDir:2.74_m:deg,
#temperature:4.61_m:C,
#temperature:1.68_m:C,
#temperature:2.75_m:C,
#temperature:3.92_m:C,
#air_water_vapor_mixing_ratio:4.61:k/kg,
#air_water_vapor_mixing_ratio:1.68:k/kg,
#air_water_vapor_mixing_ratio:2.75:k/kg,
#air_water_vapor_mixing_ratio:3.92:k/kg,
#temperature:2.87:C,
#air_water_vapor_mixing_ratio:2.87:k/kg,
#air_static_pressure:4.61_m:hPa,
#sea_surface_temperature:0_m:C,
#wu_component_flux:4.61_m:m^2/s^2,
#wv_component_flux:4.61_m:m^2/s^2,
#turbulent_heat_flux:4.61_m:m^2/s^2,
#turbulent_water_vapor_flux:4.61:g/kg m/s,
#wave_swell_time_period:0_m:s,
#wave_swell_phase_speed:0_m:m/s,
#wave_swell_energy:0_m:m^2,
#wave_swell_dir:0_m:deg,
#wave_sea_time_period:0_m:s,
#wave_sea_phase_speed:0_m:m/s,
#wave_sea_energy:0_m:m^2,
#wave_sea_dir:0_m:deg,wave_height:0_m:m
# Research Vessel Sally Ride Data

# lat lon of 36.700N 122.343W buoy 46114 off coast of Monteray, CA
# Fairly central to the data gathering area of the CASPER West project
def process_rvsr_data(csv_path, out_file, nan_column="", rvsr_lon=-122.343, rvsr_lat=36.700,
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
    print(raw_data.columns) 
    #
    # Filter out data based on "bad" data in nan_columns
    #
    #raw_data = raw_data.loc[~pd.isna(raw_data[nan_column])]
   
    #
    # List data columns included in training dataset
    #
    derived_columns = ["zenith:0_m:degrees",
                       "azimuth:0_m:degrees",
                       "temperature:1.68_m:K",
                       "temperature:2.75_m:K",
                       "temperature:3.92_m:K",  
                       "temperature:4.61_m:K",
                       "water_sfc_temperature:0_m:K",
                       "pressure:4.61_m:hPa",
                       "pressure:0_m:hPa",
                       "potential_temperature:4.61_m:K",
                       "skin_virtual_potential_temperature:0_m:K",
                       "mixing_ratio:0_m:g_kg-1",
                       "mixing_ratio:1.68_m:g_kg-1",
                       "mixing_ratio:2.75_m:g_kg-1",
                       "mixing_ratio:2.87_m:g_kg-1",
                       "mixing_ratio:3.92_m:g_kg-1",
                       "mixing_ratio:4.61_m:g_kg-1",
                       #"relative_humidity:12.54:%",
                       "wave_direction:0_m:degrees",
                       "wave_height:0_m:m",
                       "wave_period:0_m:s",
                       "wave_phase_speed:0_m:m_s-1",
                       "wave_time_period:0_m:s",
                       "wave_energy:0_m:m^2",
                       "swell_direction:0_m:degrees",
                       "swell_height:0_m:m",
                       "swell_period:0_m:s",
                       "swell_phase_speed:0_m:m_s-1",
                       "swell_energy:0_m:m^2",
                       "wind_speed:1.4_m:m_s-1",
                       "wind_speed:2.74_m:m_s-1",
                       "wind_speed:4.61_m:m_s-1",
                       "wind_direction:1.4_m:degrees",
                       "wind_direction:2.74_m:degrees",
                       "u_wind:1.4_m:m_s-1",
                       "v_wind:2.74_m:m_s-1",
                       "u_wave:0_m:m_s-1",
                       "v_wave:0_m:m_s-1",
                       "angle_between_wind_wave:0_m:degrees",
                       "angle_between_wind_swell:0_m:degrees",
                       "bulk_richardson:4.61_m:none",
                       "surface_roughness_charnock:0_m:m",
                       "surface_roughness_drennan:0_m:m",
                       "u_w:4.61_m:m2_s-2",
                       "v_w:4.61_m:m2_s-2",
                       "friction_velocity:4.61_m:m_s-1",
                       "turbulent_heat_flux:4.61_m:m2_s-2",
                       "turbulent_water_vapor_flux:4.61:g_kg-1_ m_s-1s",
                       "kinematic_sensible_heat_flux:4.61_m:K_m_s-1",
                       "temperature_scale:_m:K"
                       ]

    print( "Calculating derived variables")

    #
    # Define the derived_data dataframe
    #
    derived_data = pd.DataFrame(index=raw_data.index, columns=derived_columns, dtype=float)

    #
    # Fill in solar angles
    #
    solar_data = get_solarposition(raw_data.index, rvsr_lat, rvsr_lon, altitude=elevation, method="nrel_numba")
    derived_data["zenith:0_m:degrees"] = solar_data["azimuth"]
    derived_data["azimuth:0_m:degrees"] = solar_data["zenith"]

    #
    # Water surface temperature
    #
    #sea_surface_temperature:0_m:C
    derived_data["water_sfc_temperature:0_m:K"] = celsius_to_kelvin(raw_data["sea_surface_temperature:0_m:C"])

    #
    # Wave direction , height, period, phase speed, energy
    #
    derived_data["wave_direction:0_m:degrees"] = raw_data["wave_sea_dir:0_m:deg"]
    derived_data["wave_height:0_m:m"] = raw_data["wave_height:0_m:m"]
    derived_data["wave_period:0_m:s"] = raw_data["wave_sea_time_period:0_m:s"]  
    derived_data["wave_phase_speed:0_m:m_s-1"] = raw_data["wave_sea_phase_speed:0_m:m/s"]
    derived_data["wave_energy:0_m:m2"] = raw_data["wave_sea_energy:0_m:m^2"]

    #
    # Swell direction , period, phase speed, energy
    #
    derived_data["swell_direction:0_m:degrees"] = raw_data["wave_swell_dir:0_m:deg"]
    derived_data["swell_period:0_m:s"] = raw_data["wave_swell_time_period:0_m:s"]                                     
    derived_data["swell_phase_speed:0_m:m_s-1"] = raw_data["wave_swell_phase_speed:0_m:m/s"]
    derived_data["swell_energy:0_m:m2"] = raw_data["wave_swell_energy:0_m:m^2"]

    #
    # Wind Speed 
    #
    derived_data["wind_speed:1.4_m:m_s-1"] = raw_data["WindSpeed:1.4_m:m/s"]
    derived_data["wind_speed:2.74_m:m_s-1"] = raw_data["WindSpeed:2.74_m:m/s"]

    #
    # Wind Direction
    #
    derived_data["wind_direction:1.4_m:degrees"] = raw_data["WindDir:1.4_m:deg"]
    derived_data["wind_direction:2.74_m:degrees"] = raw_data["WindDir:2.74_m:deg"]

    #
    # Derived data wind components
    #
    derived_data["u_wind:1.4_m:m_s-1"], derived_data["v_wind:1.4_m:m_s-1"] = wind_components(derived_data["wind_speed:1.4_m:m_s-1"], derived_data["wind_direction:1.4_m:degrees"])
 
    derived_data["u_wind:2.74_m:m_s-1"], derived_data["v_wind:2.74_m:m_s-1"] = wind_components(derived_data["wind_speed:2.74_m:m_s-1"], derived_data["wind_direction:2.74_m:degrees"])

    #
    # Derived wave components
    #
    derived_data["u_wave:0_m:m_s-1"], derived_data["v_wave:0_m:m_s-1"] = wind_components(derived_data["wave_phase_speed:0_m:m_s-1"], derived_data["wave_direction:0_m:degrees"])

    derived_data["u_swell:0_m:m_s-1"], derived_data["v_swell:0_m:m_s-1"] = wind_components(derived_data["swell_phase_speed:0_m:m_s-1"], derived_data["swell_direction:0_m:degrees"])

    derived_data["angle_between_wind_wave:0_m:degrees"] = 180/np.pi * np.arccos((derived_data["u_wave:0_m:m_s-1"] * derived_data["u_wind:2.74_m:m_s-1"] + derived_data["v_wave:0_m:m_s-1"] * derived_data["v_wind:2.74_m:m_s-1"])/(derived_data["wave_phase_speed:0_m:m_s-1"] * derived_data["wind_speed:2.74_m:m_s-1"]))

    derived_data["angle_between_wind_swell:0_m:degrees"] = 180/np.pi * np.arccos((derived_data["u_swell:0_m:m_s-1"] * derived_data["u_wind:1.4_m:m_s-1"] + derived_data["v_swell:0_m:m_s-1"] * derived_data["v_wind:1.4_m:m_s-1"])/(derived_data["swell_phase_speed:0_m:m_s-1"] * derived_data["wind_speed:1.4_m:m_s-1"]))

    #
    # Pressure
    # 
    derived_data["pressure:4.61_m:hPa"] = raw_data["air_static_pressure:4.61_m:hPa"]

    #    
    # Temperature  
    #
    derived_data["temperature:1.68_m:K"] = celsius_to_kelvin(raw_data["temperature:1.68_m:C"])
    derived_data["temperature:2.75_m:K"] = celsius_to_kelvin(raw_data["temperature:2.75_m:C"])
    derived_data["temperature:3.92_m:K"] = celsius_to_kelvin(raw_data["temperature:3.92_m:C"])
    derived_data["temperature:4.61_m:K"] = celsius_to_kelvin(raw_data["temperature:4.61_m:C"])
    
    #
    # flux components
    #
    derived_data["u_w:4.61_m:m2_s-2"] = raw_data["wu_component_flux:4.61_m:m^2/s^2"]
    derived_data["v_w:4.61_m:m2_s-2"] = raw_data["wv_component_flux:4.61_m:m^2/s^2"]


    #    
    # Relative humidity 
    #
    #derived_data["relative_humidity:12_m:%"]=  raw_data['RH:12_m:%']
   
    #
    # Sea Surface/ Skin  mixing ratio (RH = 100%)
    # Note: 4.61 [m] * 9.81 [m/s^2] * 1.293 [kg/m^3] = 58.47 Pa 
    derived_data["pressure:0_m:hPa"] = derived_data["pressure:4.61_m:hPa"] + 58.47/100
    derived_data["mixing_ratio:0_m:g_kg-1"] = mixing_ratio(raw_data["sea_surface_temperature:0_m:C"], 100,  derived_data["pressure:0_m:hPa"])
    
    derived_data["mixing_ratio:1.68_m:g_kg-1"] = raw_data["air_water_vapor_mixing_ratio:1.68:k/kg"]
    derived_data["mixing_ratio:2.75_m:g_kg-1"] = raw_data["air_water_vapor_mixing_ratio:2.75:k/kg"]
    derived_data["mixing_ratio:3.92_m:g_kg-1"] = raw_data["air_water_vapor_mixing_ratio:3.92:k/kg"]
    derived_data["mixing_ratio:4.61_m:g_kg-1"] = raw_data["air_water_vapor_mixing_ratio:4.61:k/kg"]
    

    #
    # Virtual potential skin temperature : use sea surface temp
    #
    derived_data[ "skin_virtual_potential_temperature:0_m:K"] = virtual_temperature( derived_data["water_sfc_temperature:0_m:K"], derived_data["mixing_ratio:0_m:g_kg-1"])


    #
    # Derive potential temperature   
    derived_data["potential_temperature:4.61_m:K"] = potential_temperature(derived_data["temperature:4.61_m:K"], derived_data[f"pressure:4.61_m:hPa"])

    #
    # Friction Velocity: derived from u*=(〈u'w'〉^2+〈v'w'〉^2)^1/4
    #
    derived_data["u_w:4.61_m:m2_s-2"] = raw_data["wu_component_flux:4.61_m:m^2/s^2"]

    derived_data["v_w:4.61_m:m2_s-2"] = raw_data["wv_component_flux:4.61_m:m^2/s^2"]

    derived_data["friction_velocity:4.61_m:m_s-1"]= ((derived_data["u_w:4.61_m:m2_s-2"])**2 +  (derived_data["v_w:4.61_m:m2_s-2"])**2 )**(.25)

    #
    # heat flux and temperature scale
    #
    derived_data["kinematic_sensible_heat_flux:4.61_m:K_m_s-1"] = raw_data["turbulent_heat_flux:4.61_m:m^2/s^2"]

    derived_data["temperature_scale:4.61_m:K"] = derived_data["kinematic_sensible_heat_flux:4.61_m:K_m_s-1"]/derived_data["friction_velocity:4.61_m:m_s-1"]

    #
    # Water vapor flux
    #
    derived_data["water_vapor_flux:4.61:g_kg-1_m_s-1"] = raw_data["turbulent_water_vapor_flux:4.61:g/kg m/s"]

    derived_data["moisture_scale:none:none"] = derived_data["water_vapor_flux:4.61:g_kg-1_m_s-1"]/derived_data["friction_velocity:4.61_m:m_s-1"]
    
    #
    # define surface roughness as a functio of friction velocity, wave height , and wave phase speed
    # http://waveworkshop.org/13thWaves/Papers/COWCLIP_paper.pdf
    #search "Drennan et al. (2003)"
    #z0 = 3.35 * derived_data["wave_height:0_m:m"] * (derived_data["friction_velocity:18.4_m:m_s-1"]/derived_data["wave_phase_speed:0_m:m_s-1"] )**3.4
    derived_data["surface_roughness_drennan:0_m:m"] = 3.35 * derived_data["wave_height:0_m:m"] * (derived_data["friction_velocity:4.61_m:m_s-1"]/derived_data["wave_phase_speed:0_m:m_s-1"] )**3.4

    # Charnock's relation
    #z0 =  αc u*2/g 
    derived_data["surface_roughness_charnock:0_m:m"] = .015/9.8 * derived_data["friction_velocity:4.61_m:m_s-1"]**2

    #
    # d = Zero-plane displacement is the height in meters above the ground at which zero mean wind speed 
    # is achieved as a result of flow obstacles such as trees or buildings.
    #
    d = derived_data["wave_height:0_m:m"] 
    #d = 0
    #
    # https://en.wikipedia.org/wiki/Log_wind_profile
    # 
    z0 = derived_data["surface_roughness_drennan:0_m:m"]
    derived_data[ "wind_speed:4.61_m:m_s-1"] = derived_data[ "wind_speed:2.74_m:m_s-1"] * np.log((4.61 - d )/z0)/np.log((2.74 - d)/z0);

    #
    # Bulk Richardson's number  
    #  
    derived_data[ "bulk_richardson:4.61_m:none"] = bulk_richardson_number( derived_data["potential_temperature:4.61_m:K"], 4.61,
                                                                         derived_data["mixing_ratio:4.61_m:g_kg-1"],
                                                                         derived_data["skin_virtual_potential_temperature:0_m:K"],
                                                                         derived_data["wind_speed:4.61_m:m_s-1"])

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
    print( "Writing ", out_file)
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
    testDays=[4, 8, 12, 16, 20, 24, 28]
    
    data["test"] = all_data.loc[all_data.index.day.isin(testDays)]
    data["train"] = all_data.loc[all_data.index.difference(data["test"].index) ]


    train = pd.DataFrame()
    train = all_data.loc[all_data.index.isocalendar().week.isin(testWeeks)]
    train.to_csv("/Volumes/SuesRoo/mlsurfacelayer/wcoastData/csv/rvsr_train.csv", na_rep = '?')
    test = pd.DataFrame()
    test = all_data.loc[all_data.index.difference(data["test"].index) ]
    test.to_csv("/Volumes/SuesRoo/mlsurfacelayer/wcoastData/csv/rvsr_test.csv", na_rep = '?')

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


