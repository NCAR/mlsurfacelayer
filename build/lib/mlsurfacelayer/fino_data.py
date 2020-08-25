import pandas as pd
from glob import glob
from os.path import join
from .derived import *
from pvlib.solarposition import get_solarposition
import datetime

def process_fino_data(csv_path, out_file, nan_column=("Value_Surface_Temperature:Buoy_m:C"), fino1_lat=54.0, fino1_lon=6.35,
                        elevation=0.0, reflect_counter_gradient=False, average_period=None):
    """
    This function loads all of the FINO1 data fuke and then calculates the relevant derived quantities necessary
    to build the machine learning parameterization.

    Columns in derived data show follow the following convention "name_of_variable:level:units". Colons separate
    different components, and underscores separate words in each subsection. `df.columns.str.split(":").str[0]` extracts
    the variable names, `df.columns.str.split(":").str[1]` extracts levels, and `df.columns.str.split(":").str[2]`
    extracts units.

    Args:
        csv_path: Path to all csv files.
        out_file: Where derived data are written to.
        nan_column: Column used to filter bad examples.
        fino1_lat: Latitude of tower site in degrees.
        fino1_lon: Longitude of tower site in degrees.
        elevation: Elevation of site in meters.
        reflect_counter_gradient: Change the sign of counter gradient sensible and latent heat flux values.
        average_period: Window obs are averaged over.
    Returns:
        `pandas.DataFrame` containing derived data.
    """
    #
    # Read raw data
    #
    print("Reading raw data")
    csv_file = glob(join(csv_path, "*.csv"))
    print (csv_file)
    raw_data = pd.read_csv(csv_file[0], na_values=[-9999.0])

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
    derived_columns = ["global_horizontal_irradiance:30_m:W_m-2",
                       "zenith:0_m:degrees",
                       "azimuth:0_m:degrees",
#                       "temperature:30_m:K", 
                       "temperature:40_m:K",
                       "temperature:50_m:K",
                       "temperature:60_m:K",
                       "temperature:80_m:K",
                       "water_sfc_temperature:0_m:K", 
                       "pressure:0_m:hPa",
                       "pressure:30_m:hPa",
                       "pressure:40_m:hPa",
                       "pressure:50_m:hPa",
                       "pressure:60_m:hPa",
                       "pressure:80_m:hPa",
#                      "potential_temperature:30_m:K",
                       "potential_temperature:40_m:K",
                       "potential_temperature:50_m:K",
                       "potential_temperature:60_m:K",
                       "potential_temperature:80_m:K",
                       "skin_virtual_potential_temperature:0_m:K",
                       "mixing_ratio:0_m:g_kg-1",
                       "mixing_ratio:40_m:g_kg-1",
                       "mixing_ratio:60_m:g_kg-1",
                       "mixing_ratio:80_m:g_kg-1",
                       "relative_humidity:40_m:%",
                       "relative_humidity:60_m:%",
                       "relative_humidity:80_m:%",
                       "wave_dir_linear_interp:0_m:degrees",
                       "wave_dir_nearest_interp:0_m:degrees",
                       "wave_height:0_m:m",
                       "wave_period:0_m:s",
                       "wave_phase_speed:0_m:m_s-1",
                       #"wave_u:0_m:m_s-1",
                       #"wave_v:0_m:m_s-1",
                       "wind_speed:40_m:m_s-1",
                       "wind_speed:60_m:m_s-1",
                       "wind_speed:80_m:m_s-1",
                       "wind_direction:40_m:degrees",
                       "wind_direction:60_m:degrees",
                       "wind_direction:80_m:degrees",
                       "u_wind:40_m:m_s-1",
                       "v_wind:40_m:m_s-1",
                       "u_wind:60_m:m_s-1",
                       "v_wind:60_m:m_s-1",
                       "u_wind:80_m:m_s-1",
                       "v_wind:80_m:m_s-1", 
                       "bulk_richardson:40_m:none",
                       "bulk_richardson:60_m:none",
                       "bulk_richardson:80_m:none",
                       "potential_temperature_gradient:20_m:K_m-1",
                       "potential_temperature_gradient:40_m:K_m-1",
                       "wind_speed_gradient:20_m:s-1",
                       "wind_speed_gradient:40_m:s-1",
                       "u_w:40_m:m2_s-2",
                       "u_w:60_m:m2_s-2",
                       "u_w:80_m:m2_s-2",
                       "v_w:40_m:m2_s-2",
                       "v_w:60_m:m2_s-2",
                       "v_w:80_m:m2_s-2",
                       "friction_velocity:40_m:m_s-1",
                       "friction_velocity:60_m:m_s-1",
                       "friction_velocity:80_m:m_s-1",
                       "sensible_heat_flux:40_m:W_m-2", 
                       "temperature_scale:40_m:K"
                       ]

    print( "Calculating derived variables")

    #
    # Define the derived_data dataframe
    #
    derived_data = pd.DataFrame(index=raw_data.index, columns=derived_columns, dtype=float)

    #
    # Fill in solar angles
    #
    solar_data = get_solarposition(raw_data.index, fino1_lat, fino1_lon, altitude=elevation, method="nrel_numba")
    derived_data["zenith:0_m:degrees"] = solar_data["zenith"]
    derived_data["azimuth:0_m:degrees"] = solar_data["azimuth"]

    #
    # GHI
    # 
    derived_data["global_horizontal_irradiance:30_m:W_m-2"] = raw_data["pyrano_mean:30_m:w/m^2" ]

    #
    # Water surface temperature
    #
    derived_data["water_sfc_temperature:0_m:K"] = celsius_to_kelvin(raw_data["Value_Surface_Temperature:Buoy_m:C"])

    #
    # Wave direction , height, period
    #
    derived_data["wave_dir_linear_interp:0_m:degrees"] = raw_data["Value_wave_dir_linear_interp:Buoy:degree"]
    derived_data["wave_dir_nearest_interp:0_m:degrees"] = raw_data["Value_wave_dir_nearest_interp:Buoy:degree"]
    derived_data["wave_height:0_m:m"] = raw_data["Value_wave_height:Buoy:m"]
    derived_data["wave_period:0_m:s"] = raw_data["Value_wave_period_Mean:Buoy:s"]
    derived_data["wave_phase_speed:0_m:m_s-1"] = derived_data["wave_period:0_m:s"]* 9.8/(2*np.pi)
    #derived_data["wave_u:0_m:m_s-1"] =  -derived_data["wave_phase_speed:0_m:m_s-1"] * np.sin(2* np.pi * derived_data["wave_dir:0_m:degrees"])  
    #derived_data["wave_v:0_m:m_s-1"] =  -derived_data["wave_phase_speed:0_m:m_s-1"] * np.cos(2 * np.pi * derived_data["wave_dir:0_m:degrees"])

    #
    # Copy raw variables at 40,60 80m 
    #
    for height in [40, 60, 80]:
        #
        # Wind Speed 
        #
        derived_data[f"wind_speed:{height:d}_m:m_s-1"] = raw_data[f"WindSpeed:{height:d}_m:s^-1"]

        #
        # Wind Direction
        #
        derived_data[f"wind_direction:{height:d}_m:degrees"] = raw_data[f"WindDir:{height:d}_m:degree"]

        #
        # Derived data wind components
        #
        derived_data[f"u_wind:{height:d}_m:m_s-1"], derived_data[f"v_wind:{height:d}_m:m_s-1"] = wind_components(derived_data[f"wind_speed:{height:d}_m:m_s-1"], derived_data[f"wind_direction:{height:d}_m:degrees"]) 
       

    for height in [30,40,50,60,80]:
        #
        # Pressure
        # 
        derived_data[f"pressure:{height:d}_m:hPa"] = raw_data[f"Pressure:{height:d}_m:Pa"]/100

    #    
    # Temperature at 30,40,50,60,80m.  For 60 and 80m raw data interpolate linearly using measured values at 50 and 70m. 
    #
    #for height in [30,40,50]:  Note that T30 may have probs
    for height in [40,50]:
        derived_data[f"temperature:{height:d}_m:K"] =  celsius_to_kelvin(raw_data[f"Temp:{height:d}_m:C"])
    
    derived_data["temperature:60_m:K"] = celsius_to_kelvin( (raw_data['Temp:50_m:C'] + raw_data['Temp:70_m:C'] )/2 )
    derived_data["temperature:80_m:K"] = celsius_to_kelvin(raw_data['Temp:70_m:C'] + (raw_data['Temp:70_m:C'] - raw_data['Temp:50_m:C'] )/2 )

    #
    # flux components
    #
    derived_data["u_w:40_m:m2_s-2"] = raw_data["u_w:40_m:m2_s-2"]
    derived_data["u_w:60_m:m2_s-2"] = raw_data["u_w:60_m:m2_s-2"]
    derived_data["u_w:80_m:m2_s-2"] = raw_data["u_w:80_m:m2_s-2"]
    derived_data["v_w:40_m:m2_s-2"] = raw_data["v_w:40_m:m2_s-2"]
    derived_data["v_w:60_m:m2_s-2"] = raw_data["v_w:60_m:m2_s-2"]
    derived_data["v_w:80_m:m2_s-2"] = raw_data["v_w:80_m:m2_s-2"]


    #    
    # Relative humidity at 30,40,50,60,80m.  Interpolate linearly using measured values at 30 and 90m. 
    #
    derived_data["relative_humidity:30_m:%"]=  raw_data['hygro_mean:30_m:%']
    derived_data["relative_humidity:40_m:%"] = raw_data['hygro_mean:30_m:%'] + (raw_data['hygro_mean:90_m:%'] - raw_data['hygro_mean:30_m:%'])/6 
    derived_data["relative_humidity:50_m:%"] = raw_data['hygro_mean:30_m:%'] + (raw_data['hygro_mean:90_m:%'] - raw_data['hygro_mean:30_m:%'])/3
    derived_data["relative_humidity:60_m:%"] = raw_data['hygro_mean:30_m:%'] + (raw_data['hygro_mean:90_m:%'] - raw_data['hygro_mean:30_m:%'])/2
    derived_data["relative_humidity:80_m:%"] = raw_data['hygro_mean:90_m:%'] - (raw_data['hygro_mean:90_m:%'] - raw_data['hygro_mean:30_m:%'])/6
    
    #
    # Pressure at surface : linearly interpolate using pressure at 30 and 80m 
    #     
    derived_data["pressure:0_m:hPa"] = (raw_data['Pressure:80_m:Pa'] - (raw_data['Pressure:80_m:Pa'] - raw_data['Pressure:30_m:Pa'])*8/5)/100 

    #
    # Sea Surface/ Skin  mixing ratio (RH = 100%)
    #
    derived_data["mixing_ratio:0_m:g_kg-1"] = mixing_ratio(raw_data["Value_Surface_Temperature:Buoy_m:C"], 100,  derived_data["pressure:0_m:hPa"])

    #
    # Virtual potential skin temperature : use sea surface temp
    #
    derived_data[ "skin_virtual_potential_temperature:0_m:K"] = virtual_temperature( derived_data["water_sfc_temperature:0_m:K"], derived_data["mixing_ratio:0_m:g_kg-1"])

      
    #
    # Derive potential temperature 
    #
    #for height in [30,40,50,60,80]:
    for height in [40,50,60,80]:
        #
        # Potential Temperature
        #
        derived_data[f"potential_temperature:{height:d}_m:K"] = potential_temperature(derived_data[f"temperature:{height:d}_m:K"], derived_data[f"pressure:{height:d}_m:hPa"])

    #
    # Derive variables at 40,60 80m 
    #
    for height in [40,60,80]:
        #
        # Mixing ratio
        #
        derived_data[f"mixing_ratio:{height:d}_m:g_kg-1"] = mixing_ratio( derived_data[f"temperature:{height:d}_m:K"]-273, derived_data[f"relative_humidity:{height:d}_m:%"], derived_data[f"pressure:{height:d}_m:hPa"])

        #
        # Bulk Richardson's number
        #  
        derived_data[ f"bulk_richardson:{height:d}_m:none"] = bulk_richardson_number( derived_data[f"potential_temperature:{height:d}_m:K"], height, 
                                                                                      derived_data[f"mixing_ratio:{height:d}_m:g_kg-1"], 
                                                                                      derived_data["skin_virtual_potential_temperature:0_m:K"], 
                                                                                      derived_data[f"wind_speed:{height:d}_m:m_s-1"])
    #
    # Wind speed and potential temperature gradients
    #    
    heights = [40,60,80]
    for dh, diff_height in enumerate([20,40]):
        derived_data[f"wind_speed_gradient:{diff_height:d}_m:s-1"] = (derived_data[f"wind_speed:{heights[dh+1]:d}_m:m_s-1"] -
                                                                      derived_data[f"wind_speed:{heights[dh]:d}_m:m_s-1"]) /(heights[dh+1] - heights[dh])

        derived_data[f"potential_temperature_gradient:{diff_height:d}_m:K_m-1"] = (derived_data[f"potential_temperature:{heights[dh+1]:d}_m:K"] -
                                                                                 derived_data[f"potential_temperature:{heights[dh]:d}_m:K"]) /(heights[dh+1] - heights[dh])
    #
    # Friction Velocity: 40 (given in raw data), 60 and 80m (derived from u*=(〈u'w'〉^2+〈v'w'〉^2)^1/4
    #
    derived_data["friction_velocity:40_m:m_s-1"] = raw_data["ustar:40_m:u'w'"]
 
    for height in [60,80]:                                                
        derived_data[f"friction_velocity:{height:d}_m:m_s-1"]= ((raw_data[f'u_w:{height:d}_m:m2_s-2'])**2 +  (raw_data[f'v_w:{height:d}_m:m2_s-2'])**2 )**(.25)    

    derived_data["sensible_heat_flux:40_m:W_m-2"] = raw_data["Sensible Heat Flux:40_m:K_m_s-1"]

    derived_data["temperature_scale:40_m:K"] = derived_data["sensible_heat_flux:40_m:W_m-2"]/derived_data["friction_velocity:40_m:m_s-1"]

    #
    # Create rolling average of data columns if requested
    #
    if average_period is not None:
        derived_data = derived_data.rolling(window=average_period).mean()
        derived_data = derived_data.dropna()

    #
    # Output data
    #
    derived_data = derived_data.dropna()
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

def load_derived_data_random_test_train(filename, dropna=True, filter_counter_gradient=False): 
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
    if dropna:
        all_data = all_data.dropna()
    if filter_counter_gradient:
        all_data = filter_counter_gradient_data(all_data)
    data = dict()

    #
    # For repeatability we use the same set of randomly generated weeks of the 
    # year for testing. The compliment set is for training.
    #
    randomWeeks2010=[4, 8, 10, 15, 19, 23, 27, 29, 33, 39, 43, 47, 52]
    randomWeeks2006=[3, 8, 9, 14, 20, 21, 27, 31, 35, 38, 43, 45, 51]


    #
    # For every month of data, choose a random week for testing and 
    # leave the rest for trainin
    #
    
    data["test"] = all_data.loc[(all_data.index.weekofyear.isin(randomWeeks2010)&(all_data.index.year == 2010)) |
                                (all_data.index.weekofyear.isin(randomWeeks2006)&(all_data.index.year == 2006))  ]
    data["train"] = all_data.loc[all_data.index.difference(data["test"].index) ]

    train = pd.DataFrame()
    train = all_data.loc[(all_data.index.weekofyear.isin(randomWeeks2010)&(all_data.index.year == 2010)) |
                                (all_data.index.weekofyear.isin(randomWeeks2006)&(all_data.index.year == 2006))  ]
    train.to_csv("/d1/FINO1/cubist_fino_train.csv", na_rep = '?')
    test = pd.DataFrame()
    test = all_data.loc[all_data.index.difference(data["test"].index) ]
    test.to_csv("/d1/FINO1/cubist_fino_test.csv", na_rep = '?')
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

