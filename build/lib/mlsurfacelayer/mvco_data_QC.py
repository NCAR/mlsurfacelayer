import pandas as pd
from glob import glob
from os.path import join
from .derived import *
from mlsurfacelayer import mo
from pvlib.solarposition import get_solarposition
import datetime
from sklearn.utils import shuffle

'''
This file will process the QC dataset, so that the inputs to the ML model are part non-QC and part QC variables.
/glade/work/brummet/Oracle/data/MVCO/processed/qced_MVCO_ocn_sonic_vaisala_QC.20010619-20160610.outer_join.csv
# Vars of interest
DateTime,
wave_height:0_m:m, wave_period:0_m:deg, wave_dir:0_m:deg, 
w:18.4_m:m/s, 
uv_3D1:18.4_m:m2/s2, uw_3D1:18.4_m:m2/s2, vw_3D1:18.4_m:m2/s2, wT_3D1:18.4_m:m/s K, 

# wind components
u1_QC, v1_QC,
u2_QC, v2_QC,

# UW covariance ? (reported as momentum flux, MVCO QC paper
# references conversion of u,v to vector in direction of wind
# which could explain )
UpWpBar1_QC, UpWpBar2_QC,

# Heat flux
WpTpBar1_QC, WpTpBar2_QC,
 
# air temp, humidity, pressure at 18.4m  
AT_QC, RH_QC, P_QC, 

#sea surface temp
SST_QC,

# current direction components
uc_QC, vc_QC,

# Short wave and long wave radiation
SW_QC, 
LW_QC, LW_QC, LW_QC, LW_QC, why 4 of these?

'''

def process_mvco_data(csv_path, out_file, nan_column="", mvco_lon=-70.544, mvco_lat=41.3435,
                        elevation=0.0, average_period=None):
    """
    This function loads all of the MVCO data and then calculates the relevant derived quantities necessary
    to build the machine learning model for parameterization of surface layer momentum and heat fluxes.

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
    raw_data.index = pd.to_datetime(raw_data["DateTime"], errors='coerce' )#format="%m/%d/%y %H:%M:%m") 

    verbose = 2  
    # if verbose = 2, debugging print statements will activate
    # if verbose = 1, general info to be printed that user might want to know
    # if verbose = 0, do nothing

    #
    # List data columns included in training dataset
    #
    derived_columns = ["zenith:0_m:degrees",
                       "azimuth:0_m:degrees",
                       "GHI:0_m:W_m-2",
                       "pressure:18.4_m:hPa",
                       "sea_surface_pressure:0_m:hPa",
                       "relative_humidity:18.4_m:%",
                       "temperature:18.4_m:K",
                       "potential_temperature:18.4_m:K", 
                       "water_sfc_temperature:0_m:K",
                       "mixing_ratio:0_m:g_kg-1",
                       "mixing_ratio:18.4_m:g_kg-1",
                       "skin_virtual_potential_temperature:0_m:K",
                       "wave_direction:0_m:degrees",
                       "wave_height:0_m:m",
                       "wave_period:0_m:s",
                       "wave_phase_speed:0_m:m_s-1",
                       "u_wave:0_m:m_s-1",
                       "v_wave:0_m:m_s-1",
                       "near_surf_current_u:0_m:m_s-1",
                       "near_surf_current_v:0_m:m_s-1",
                       "near_surf_current:0_m:m_s-1",
                       "near_surf_current_dir:0_m:deg",
                       "u_wind:18.4_m:m_s-1",
                       "v_wind:18.4_m:m_s-1",
                       "wind_speed:18.4_m:m_s-1",
                       "wind_direction:18.4_m:degrees",
                       "angle_between_wind_wave:0_m:degrees",
                       "bulk_richardson:18.4_m:none",
                       "dT_dz:18.4_m:K_m-1",
                       "dSpeed_dz:18.4_m:s-1",
                       "u_w:18.4_m:m2_s-2",
                       "v_w:18.4_m:m2_s-2",
                       "T_w:18.4_m:C_m_s-2",
                       "heat_flux:18.4_m:degrees_K_m_s-1",
                       "momentum_flux:18.4_m:m2_s-2",
                       "friction_velocity:18.4_m:m_s-1",
                       "temperature_scale:18.4_m:K",
                       "MOST_momentum_flux:18.4_m:m2_s-2",
                       "MOST_heat_flux:18.4_m:degrees_K_m_s-1",
                       "MOST_chopped_momentum_flux:18.4_m:m2_s-2",
                       "MOST_chopped_heat_flux:18.4_m:degrees_K_m_s-1",
                       "MOST_rounded_momentum_flux:18.4_m:m2_s-2",
                       "MOST_rounded_heat_flux:18.4_m:degrees_K_m_s-1"]



    print( "Calculating derived variables")

    #
    # Define the derived_data dataframe
    #
    if verbose == 2: print("Define the derived_data dataframe: done")
    derived_data = pd.DataFrame(index=raw_data.index, columns=derived_columns, dtype=float)

    #
    # Fill in solar angles
    #
    if verbose == 2: print("Fill in solar angles")
    solar_data = get_solarposition(raw_data.index, mvco_lat, mvco_lon, altitude=elevation, method="nrel_numba")
    derived_data["zenith:0_m:degrees"] = solar_data["zenith"]
    derived_data["azimuth:0_m:degrees"] = solar_data["azimuth"]

    #
    # Global Horizontal Irradiance GHI
    #
    derived_data["GHI:0_m:W_m-2"] = raw_data["SW_QC"]
    
    #
    # Pressure
    #
    derived_data["pressure:18.4_m:hPa"] = raw_data["P_QC"] 

    #
    # Sea Surface pressure
    # P0 = P18 + delta P, deltaP = rho*g*deltah, rho = 1.293 air density of Pure, dry air
    # Note: 18.4 [m] * 9.81 [m/s^2] * 1.293 [kg/m^3] = 233.391672  Pa
    if verbose == 2: print("Sea Surface Pressure/ Skin mixing ratio")
    derived_data["sea_surface_pressure:0_m:hPa"] = derived_data["pressure:18.4_m:hPa"] + 233.391672/100 

    #
    # RH
    # 
    derived_data["relative_humidity:18.4_m:%"]=  raw_data['RH_QC'] 
    
    #    
    # Temperature  
    #
    if verbose == 2: print("Temperature")
    derived_data["temperature:18.4_m:K"] =  celsius_to_kelvin(raw_data["AT_QC"])     

    #
    # Potential temperature     
    #
    if verbose == 2: print("Potential temperature")
    derived_data["potential_temperature:18.4_m:K"] = potential_temperature(derived_data["temperature:18.4_m:K"], derived_data["pressure:18.4_m:hPa"])


    #
    # Water surface temperature
    #
    if verbose == 2: print("Water surface temperature")
    derived_data["water_sfc_temperature:0_m:K"] = celsius_to_kelvin(raw_data["SST_QC"]) 

    #
    # Mixing ratios
    #
    if verbose == 2: print("Skin mixing ratio")
    derived_data["mixing_ratio:0_m:g_kg-1"] = mixing_ratio(raw_data["SST_QC"], 100, derived_data["sea_surface_pressure:0_m:hPa"]) 
    derived_data["mixing_ratio:18.4_m:g_kg-1"] = mixing_ratio( derived_data["temperature:18.4_m:K"]-273, derived_data["relative_humidity:18.4_m:%"], derived_data["pressure:18.4_m:hPa"])

    #
    # skin virtual potential temperature
    #
    if verbose == 2: print("Virtual potential skin temperature : use sea surface temp")
    derived_data[ "skin_virtual_potential_temperature:0_m:K"] = virtual_temperature( derived_data["water_sfc_temperature:0_m:K"], derived_data["mixing_ratio:0_m:g_kg-1"])

    #
    # Wave direction, direction components, height, period, wave phase speed
    #
    if verbose == 2: print("Wave direction , height, period")
    derived_data["wave_direction:0_m:degrees"] = raw_data["wave_dir:0_m:deg"]
    derived_data["wave_period:0_m:s"] = raw_data["wave_period:0_m:deg"] 
    derived_data["wave_phase_speed:0_m:m_s-1"] = derived_data["wave_period:0_m:s"]* 9.8/(2*np.pi)
    derived_data["u_wave:0_m:m_s-1"] = - derived_data["wave_phase_speed:0_m:m_s-1"]*np.sin( derived_data["wave_direction:0_m:degrees"])
    derived_data["v_wave:0_m:m_s-1"] = -derived_data["wave_phase_speed:0_m:m_s-1"]*np.cos( derived_data["wave_direction:0_m:degrees"]) 
    derived_data["wave_height:0_m:m"] = raw_data["wave_height:0_m:m"]

    #
    # Current variables
    #
    if verbose == 2: print("Current variables")
    derived_data["near_surf_current_u:0_m:m_s-1"], derived_data["near_surf_current_v:0_m:m_s-1"] = raw_data["uc_QC"] /100 , raw_data["vc_QC"]/100
    derived_data["near_surf_current:0_m:m_s-1"] = np.sqrt(derived_data["near_surf_current_u:0_m:m_s-1"]**2 + derived_data["near_surf_current_v:0_m:m_s-1"]**2) 
    derived_data["near_surf_current_dir:0_m:deg"] = raw_data["near_surf_current_dir:0_m:deg"]
    

    #
    # Wind  
    #
    if verbose == 2: print("Wind vars")
    derived_data["u_wind:18.4_m:m_s-1"], derived_data["v_wind:18.4_m:m_s-1"] = raw_data["u1_QC"], raw_data["v1_QC"]
    derived_data["wind_direction:18.4_m:degrees"] = np.arctan2(derived_data["u_wind:18.4_m:m_s-1"],derived_data["v_wind:18.4_m:m_s-1"]) * 180/np.pi 
    derived_data["wind_speed:18.4_m:m_s-1"] = np.sqrt(derived_data['u_wind:18.4_m:m_s-1']**2 + derived_data['v_wind:18.4_m:m_s-1']**2)

    #
    # Angle between wind and wave
    #
    derived_data["angle_between_wind_wave:0_m:degrees"] = 180/np.pi * np.arccos((derived_data["u_wave:0_m:m_s-1"] * derived_data["u_wind:18.4_m:m_s-1"] + derived_data["v_wave:0_m:m_s-1"] * derived_data["v_wind:18.4_m:m_s-1"])/(derived_data["wave_phase_speed:0_m:m_s-1"] * derived_data["wind_speed:18.4_m:m_s-1"]))

    #
    # Bulk Richardson's number Note that wspd is at a diff height 
    #  
    if verbose == 2: print("Bulk Richardson's number")
    derived_data["bulk_richardson:18.4_m:none"]= bulk_richardson_number( derived_data["potential_temperature:18.4_m:K"], 18.4,
                                                                         derived_data["mixing_ratio:18.4_m:g_kg-1"],
                                                                         derived_data["skin_virtual_potential_temperature:0_m:K"],
                                                                         derived_data["wind_speed:18.4_m:m_s-1"])

    #
    # Derivatives: (derived_data["Var(z1)"] - derived_data["Var(z2)"]) / (z1 - z2)
    # z2 is assumed to be 0 : height at surface of water is 0
    #
    z1 = 18.4
    derived_data["dT_dz:18.4_m:K_m-1"]  = (derived_data["temperature:18.4_m:K"] - derived_data["water_sfc_temperature:0_m:K"]) / z1
    derived_data["dSpeed_dz:18.4_m:s-1"]= derived_data["wind_speed:18.4_m:m_s-1"] / z1

    #
    # flux components
    #
    if verbose == 2: print("Raw flux components")
    derived_data["u_w:18.4_m:m2_s-2"] = raw_data["uw_3D1:18.4_m:m2/s2"]
    derived_data["v_w:18.4_m:m2_s-2"] = raw_data["vw_3D1:18.4_m:m2/s2"]
    derived_data["w_T:18.4_m:K_m2_s-2"] = raw_data["wT_3D1:18.4_m:m/s K"]

    #
    # Momentum Flux & Heat Flux
    #
    if verbose == 2: print("Fluxes")
    derived_data['momentum_flux:18.4_m:m2_s-2'] = raw_data["UpWpBar1_QC"]
    derived_data['heat_flux:18.4_m:degrees_K_m_s-1'] = celsius_to_kelvin(raw_data["WpTpBar1_QC"])
    derived_data['friction_velocity:18.4_m:m_s-1'] = np.sqrt(derived_data['momentum_flux:18.4_m:m2_s-2'])
    derived_data['temperature_scale:18.4_m:K'] = 0 # put this in 

    #
    # Add MOST momentum and temp flux from inputs
    # height, Ri, skinPotT, potT, wspd, waveHt, wavePhaseSpd
    #
    if verbose == 2: print("MOST Fluxes")
    derived_data[['MOST_momentum_flux:18.4_m:m2_s-2','MOST_heat_flux:18.4_m:degrees_K_m_s-1']]  = derived_data.apply(
       lambda row: pd.Series(mo.computeMOSTfluxes(18.4, 
                                                  row['bulk_richardson:18.4_m:none'], 
                                                  row['skin_virtual_potential_temperature:0_m:K'], 
                                                  row['potential_temperature:18.4_m:K'],
                                                  row['wind_speed:18.4_m:m_s-1'],
                                                  row['wave_height:0_m:m'],
                                                  row['wave_phase_speed:0_m:m_s-1'])), axis=1) 
    
    # Perform rounding
    derived_data[['MOST_rounded_momentum_flux:18.4_m:m2_s-2','MOST_rounded_heat_flux:18.4_m:degrees_K_m_s-1']] = derived_data[['MOST_momentum_flux:18.4_m:m2_s-2','MOST_heat_flux:18.4_m:degrees_K_m_s-1']].applymap(lambda x: np.round(x, 2))

    # Perform chopping
    derived_data[['MOST_chopped_momentum_flux:18.4_m:m2_s-2','MOST_chopped_heat_flux:18.4_m:degrees_K_m_s-1']] = derived_data[['MOST_momentum_flux:18.4_m:m2_s-2','MOST_heat_flux:18.4_m:degrees_K_m_s-1']].applymap(lambda x: np.floor(x * 100) / 100)
    #momentum_flux = []
    #heat_flux = []

    #for index, row in derived_data.iterrows():
    #    fluxes = mo.computeMOSTfluxes(18.4, 
    #                                  row['bulk_richardson:18.4_m:none'],
    #                                  row['skin_virtual_potential_temperature:0_m:K'],
    #                                  row['potential_temperature:18.4_m:K'],
    #                                  row['wind_speed:18.4_m:m_s-1'],
    #                                  row['wave_height:0_m:m'],
    #                                  row['wave_phase_speed:0_m:m_s-1'])
    #    #print("fluxes: ", fluxes[0], fluxes[1])
    #    momentum_flux.append(fluxes[0])
    #    heat_flux.append(fluxes[1])

    #derived_data['MOST_momentum_flux:18.4_m:m2_s-2'] = momentum_flux
    #derived_data['MOST_heat_flux:18.4_m:degrees_K_m_s-1'] = heat_flux

    #
    # derived_data = derived_data.dropna()
    #
    if verbose == 2: print("Output data")
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
    
    # all_data = all_data.dropna(subset=[ "zenith:0_m:degrees", "azimuth:0_m:degrees", "temperature:12_m:K", "water_sfc_temperature:0_m:K", "pressure:12_m:hPa", "potential_temperature:12_m:K", "skin_virtual_potential_temperature:0_m:K", "mixing_ratio:0_m:g_kg-1", "mixing_ratio:12_m:g_kg-1", "relative_humidity:12_m:%", "wave_direction:0_m:degrees", "wave_height:0_m:m", "wave_period:0_m:s", "wave_phase_speed:0_m:m_s-1", "wind_speed:18.4_m:m_s-1", "wind_direction:18.4_m:degrees", "angle_between_wind_wave:0_m:degrees","bulk_richardson:12_m:none"])
    all_data = all_data.dropna(subset=[ 
            "zenith:0_m:degrees", 
            "azimuth:0_m:degrees", 
            # "temperature:18.4_m:K", 
            "water_sfc_temperature:0_m:K", 
            "pressure:18.4_m:hPa", 
            "potential_temperature:18.4_m:K", 
            "skin_virtual_potential_temperature:0_m:K", 
            # "mixing_ratio:0_m:g_kg-1", 
            "mixing_ratio:18.4_m:g_kg-1", 
            "relative_humidity:18.4_m:%", 
            "wave_direction:0_m:degrees", 
            "wave_height:0_m:m", 
            "wave_period:0_m:s", 
            # "wave_phase_speed:0_m:m_s-1", 
            "wind_speed:18.4_m:m_s-1", 
            "wind_direction:18.4_m:degrees", 
            "angle_between_wind_wave:0_m:degrees", 
            "bulk_richardson:18.4_m:none"])
    
    #if dropna:
    #     all_data = all_data.dropna()    
    #if filter_counter_gradient:
    #    all_data = filter_counter_gradient_data(all_data)
    data = dict()

    #
    # For repeatability we use the same set of evenly spaced test weeks  
    #
    testWeeks=[1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]

    validateWeeks = [1, 5, 9, 13, 17, 21, 25]
    
    data["test"] = all_data.loc[all_data.index.isocalendar().week.isin(testWeeks)]
    # data["validate"] = all_data.loc[all_data.index.isocalendar().week.isin(validateWeeks)]
    data["train"] = all_data.loc[all_data.index.difference(data["test"].index) ]
    
    train = pd.DataFrame()
    train = all_data.loc[all_data.index.isocalendar().week.isin(testWeeks)]
    train.to_csv("../hector/mvco_train_qc.csv", na_rep = '?')
    test = pd.DataFrame()
    test = all_data.loc[all_data.index.difference(data["test"].index)]
    test.to_csv("../hector/mvco_test_qc.csv", na_rep = '?')

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



# functins for cross validation
def split_dataframe(df, N): # Assuming df is your original DataFrame
    size = len(df) // N # Calculate the size of each smaller DataFrame
    df_list = [df.iloc[i*size:(i+1)*size] for i in range(N)] # Create a list of smaller DataFrames
    return df_list

def shift_df_list(df_list, k):
    # Shift the list over k times
    k = k % len(df_list)  # Ensure k is within the bounds of the list length
    return df_list[-k:] + df_list[:-k]

def merge_dataframes(df_list):    
    train_df = pd.concat(df_list[:-2], ignore_index=True) # Merge all but the last 2 DataFrames
    val_df = df_list[-2] # The second to last DataFrame will be the validation set
    test_df = df_list[-1] # The last DataFrame will be the test set
    return train_df, val_df, test_df

def load_derived_data_random_test_train_val(filename, dropna=False, filter_counter_gradient=False, devVar=None, N=10, k=0, scramble=False, holdout_ratio=None):
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
    # this creates a 75, 12.5, 12.5 split

    all_data = pd.read_csv(filename, index_col="Time", parse_dates=["Time"])
    all_data =  all_data[~all_data.index.duplicated(keep='first')]
    

    ''' the mf in the mvco qc data has a factor of -1 applied to it. we need to first undo by applying another factor of -1. Then we will eliminate the negative values bc mf cannot be negative (the MOST computations will always result in a positive value).
    '''
    #pd.set_option('display.max_rows', 250)
    #print('\n\n\nbefore\n',all_data['momentum_flux:18.4_m:m2_s-2'].head(250),'\n')
    #all_data = all_data[all_data['momentum_flux:18.4_m:m2_s-2'].mul(-1) >= 0]

    #print(all_data["momentum_flux:18.4_m:m2_s-2"].where(-all_data["momentum_flux:18.4_m:m2_s-2"] >0).head(250), '\n')
    all_data["momentum_flux:18.4_m:m2_s-2"] = -all_data["momentum_flux:18.4_m:m2_s-2"] # first we apply the factor -1 to switch the polarity of the values
    all_data["momentum_flux:18.4_m:m2_s-2"] = all_data["momentum_flux:18.4_m:m2_s-2"].where(all_data["momentum_flux:18.4_m:m2_s-2"] >=0) # then we seperate the positive values and leave the negative values behind

    #print('\nafter\n' ,all_data['momentum_flux:18.4_m:m2_s-2'].head(250),'\n\n\n')
    #pd.reset_option('display.max_rows')    

    
    all_data = all_data.dropna(subset=[ 
            "zenith:0_m:degrees", 
            "azimuth:0_m:degrees", 
            # "temperature:18.4_m:K", 
            "water_sfc_temperature:0_m:K", 
            "pressure:18.4_m:hPa", 
            "potential_temperature:18.4_m:K", 
            "skin_virtual_potential_temperature:0_m:K", 
            # "mixing_ratio:0_m:g_kg-1", 
            "mixing_ratio:18.4_m:g_kg-1", 
            "relative_humidity:18.4_m:%", 
            "wave_direction:0_m:degrees", 
            "wave_height:0_m:m", 
            "wave_period:0_m:s", 
            # "wave_phase_speed:0_m:m_s-1", 
            "wind_speed:18.4_m:m_s-1", 
            "wind_direction:18.4_m:degrees", 
            "angle_between_wind_wave:0_m:degrees", 
            "bulk_richardson:18.4_m:none",
            "momentum_flux:18.4_m:m2_s-2"])


    data = dict()

    #
    # For repeatability we use the same set of evenly spaced test weeks  
    #
    # testWeeks=[1, 5, 9, 13, 17, 21, 25]#29, 33, 37, 41, 45, 49]
    # validateWeeks = [29, 33, 37, 41, 45, 49]
    #
    testWeeks=[1, 33, 9, 41, 17, 49, 25]#29, 33, 37, 41, 45, 49]
    validateWeeks = [29, 5, 37, 13, 45, 21]

    if devVar == "temporal":
        testWeeks=[43, 44, 45, 46, 47, 48, 49]#29, 33, 37, 41, 45, 49]
        validateWeeks = [37, 38, 39, 40, 41, 42]
    
    data["test"] = all_data.loc[all_data.index.isocalendar().week.isin(testWeeks)]
    data["validate"] = all_data.loc[all_data.index.isocalendar().week.isin(validateWeeks)]
    data["train"] = all_data.loc[all_data.index.difference(pd.concat([pd.Series(data["test"].index), 
                                                                        pd.Series(data["validate"].index)]))]

    if scramble == True:
        all_data = shuffle(all_data, random_state=42)

    if holdout_ratio != None:
        # Calculate holdout set size
        holdout_size = int(len(all_data) * holdout_ratio)
        data["holdout"] = all_data.iloc[:holdout_size]
        print("data holdout",len(data['holdout']))
        print(f'data all {len(all_data)}')

        # Remaining Data
        all_data = all_data.iloc[holdout_size:]

        # Save holdout set to CSV
        data["holdout"].to_csv("../../data/mvco_mlsl_holdout.csv", na_rep='?')

    if devVar == "kfold":
        df_list = split_dataframe(all_data, N)              # into N smaller DataFrames
        df_list = shift_df_list(df_list, k)                 # shift df's in list to get the correct kfold
        train, validate, test = merge_dataframes(df_list)   # remerge the training and return
        data = {'train': train, 'validate': validate, 'test': test} # create dictionary to return

    if devVar == "kfold_final_model": # leave test set empty 
        df_list = split_dataframe(all_data, N)              # into N smaller DataFrames
        df_list = shift_df_list(df_list, k)                 # shift df's in list to get the correct kfold
        train, validate, test = merge_dataframes_final_model(df_list)   # remerge the training and return
        data = {'train': train, 'validate': validate, 'test': test} # create dictionary to return

    # train.to_csv("../hector/mvco_train_qc.csv", na_rep = '?')
    # validate.to_csv("../hector/mvco_validate_qc.csv", na_rep = '?')
    # test.to_csv("../hector/mvco_test_qc.csv", na_rep = '?')

    print("data loaded")
    return data


if __name__ == "__main__":
    #data = load_derived_data_random_test_train_val("../../../../data/mvco_mlsl_qc.csv")
    #print(data)
    # print(data["train"])
    # print(data["validate"])
    # print(data["test"])
    pass
