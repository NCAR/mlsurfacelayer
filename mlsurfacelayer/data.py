import pandas as pd
from glob import glob
from os.path import join
from .derived import *
from pvlib.solarposition import get_solarposition
import datetime

def process_cabauw_data(csv_path, out_file, nan_column=("soil_water", "TH03"), cabauw_lat=51.971, cabauw_lon=4.926,
                        elevation=-0.7, reflect_counter_gradient=False, average_period=None):
    """
    This function loads all of the cabauw data files and then calculates the relevant derived quantities necessary
    to build the machine learning parameterization.

    Columns in derived data show follow the following convention "name of variable_level_units". Underscores separate
    different components, and spaces separate words in each subsection. `df.columns.str.split("_").str[0]` extracts
    the variable names, `df.columns.str.split("_").str[1]` extracts levels, and `df.columns.str.split("_").str[0]`
    extracts units.

    Args:
        csv_path: Path to all csv files.
        out_file: Where derived data are written to.
        nan_column: Column used to filter bad examples.
        cabauw_lat: Latitude of tower site in degrees.
        cabauw_lon: Longitude of tower site in degrees.
        elevation: Elevation of site in meters.
        reflect_counter_gradient: Change the sign of counter gradient sensible and latent heat flux values.
        average_period: Window obs are averaged over.
    Returns:
        `pandas.DataFrame` containing derived data.
    """
    csv_files = sorted(glob(join(csv_path, "*.csv")))
    file_types = ["_".join(csv_file.split("/")[-1].split("_")[1:-1]) for csv_file in csv_files]
    data = dict()
    for c, csv_file in enumerate(csv_files):
        print(csv_file)
        data[file_types[c]] = pd.read_csv(csv_file, na_values=[-9999.0])
        data[file_types[c]].index = pd.to_datetime(data[file_types[c]]["TimeStr"], format="%Y%m%d.%H:%M")
    print("combine data")
    combined_data = pd.concat(data, axis=1, join="inner")
    combined_data = combined_data.loc[~pd.isna(combined_data[nan_column])]
    # List of columns included in data
    derived_columns = ["global horizontal irradiance_0 m_W m-2",
                       "zenith_0 m_degrees",
                       "azimuth_0 m_degrees",
                       "temperature_2 m_K",
                       "temperature_10 m_K",
                       "temperature_20 m_K",
                       "temperature_40 m_K",
                       "pressure_2 m_hPa",
                       "potential temperature_2 m_K",
                       "potential temperature_10 m_K",
                       "potential temperature_20 m_K",
                       "potential temperature_40 m_K",
                       "virtual potential temperature_2 m_K",
                       "virtual potential temperature_10 m_K",
                       "virtual potential temperature_20 m_K",
                       "virtual potential temperature_40 m_K",
                       "mixing ratio_2 m_g kg-1",
                       "mixing ratio_10 m_g kg-1",
                       "mixing ratio_20 m_g kg-1",
                       "mixing ratio_40 m_g kg-1",
                       "relative humidity_2 m_%",
                       "relative humidity_10 m_%",
                       "relative humidity_20 m_%",
                       "relative humidity_40 m_%",
                       "temperature change_4 m_K m-1",
                       "temperature change_15 m_K m-1",
                       "temperature change_30 m_K m-1",
                       "mixing ratio change_4 m_g kg-1 m-1",
                       "mixing ratio change_15 m_g kg-1 m-1",
                       "mixing ratio change_30 m_g kg-1 m-1",
                       "upward longwave irradiance_0 m_W m-2",
                       "downward longwave irradiance_0 m_W m-2",
                       "upward shortwave irradiance_0 m_W m-2",
                       "skin temperature_0 m_K",
                       "skin potential temperature_0 m_K",
                       "skin saturation mixing ratio_0 m_g kg-1",
                       "skin virtual potential temperature_0 m_K",
                       "potential temperature skin change_10 m_K m-1",
                       "potential temperature skin change_20 m_K m-1",
                       "potential temperature skin change_40 m_K m-1",
                       "virtual potential temperature skin change_10 m_K m-1",
                       "virtual potential temperature skin change_20 m_K m-1",
                       "virtual potential temperature skin change_40 m_K m-1",
                       "mixing ratio skin change_10 m_g kg-1 m-1",
                       "mixing ratio skin change_20 m_g kg-1 m-1",
                       "mixing ratio skin change_40 m_g kg-1 m-1",
                       "air density_10 m_kg m-3",
                       "wind speed_10 m_m s-1",
                       "wind direction_10 m_degrees",
                       "wind speed_20 m_m s-1",
                       "wind direction_20 m_degrees",
                       "wind speed_40 m_m s-1",
                       "wind direction_40 m_degrees",
                       "u wind_10 m_m s-1",
                       "v wind_10 m_m s-1",
                       "u wind_20 m_m s-1",
                       "v wind_20 m_m s-1",
                       "u wind_40 m_m s-1",
                       "v wind_40 m_m s-1",
                       "soil temperature_0 cm_K",
                       "soil temperature_4 cm_K",
                       "soil potential temperature_0 cm_K",
                       "soil potential temperature_4 cm_K",
                       "soil water content_3 cm_m3 m-3",
                       #"soil water content_8 cm_m3 m-3",
                       "moisture availability_3 cm_",
                       #"moisture availability_8 cm_",
                       "bulk richardson_10 m_",
                       "bulk richardson_2 m_",
                       "bulk richardson_10-2 m_",
                       "obukhov length_surface_m",
                       "sensible heat flux_surface_W m-2",
                       "latent heat flux_surface_W m-2",
                       "friction velocity_surface_m s-1",
                       "temperature scale_surface_K",
                       "soil heat flux_surface_W m-2",
                       "moisture scale_surface_g kg-1",
                       ]
    derived_data = pd.DataFrame(index=combined_data.index, columns=derived_columns, dtype=float)
    solar_data = get_solarposition(combined_data.index, cabauw_lat, cabauw_lon, altitude=elevation, method="nrel_numba")
    print("calculate derived variables")
    derived_data["global horizontal irradiance_0 m_W m-2"] = combined_data[("surface", "SWD")]
    derived_data["zenith_0 m_degrees"] = solar_data["zenith"]
    derived_data["azimuth_0 m_degrees"] = solar_data["azimuth"]
    derived_data["pressure_2 m_hPa"] = combined_data[("surface", "P0")]
    for height in [2, 10, 20, 40]:
        derived_data[f"temperature_{height:d} m_K"] = combined_data[("tower", f"TA_{height:d}m")]
        derived_data[f"mixing ratio_{height:d} m_g kg-1"] = combined_data[("tower", f"Q_{height:d}m")]
        derived_data[f"relative humidity_{height:d} m_%"] = combined_data[("tower", f"RH_{height:d}m")]

        derived_data[f"potential temperature_{height:d} m_K"] = potential_temperature(derived_data[f"temperature_{height:d} m_K"],
                                                                                      derived_data["pressure_2 m_hPa"])
        derived_data[f"virtual potential temperature_{height:d} m_K"] = virtual_temperature(derived_data[f"potential temperature_{height:d} m_K"], derived_data[f"mixing ratio_{height:d} m_g kg-1"])
    heights = [2, 10, 20, 40]
    for dh, diff_height in enumerate([4, 15, 30]):
        derived_data[f"temperature change_{diff_height:d} m_K m-1"] = (derived_data[f"temperature_{heights[dh+1]:d} m_K"] -
                                                                       derived_data[f"temperature_{heights[dh]:d} m_K"]) / (heights[dh+1] - heights[dh])
        derived_data[f"mixing ratio change_{diff_height:d} m_g kg-1 m-1"] = (derived_data[f"mixing ratio_{heights[dh+1]:d} m_g kg-1"] -
                                                                             derived_data[f"mixing ratio_{heights[dh]:d} m_g kg-1"]) / (heights[dh+1] - heights[dh])

    derived_data["virtual potential temperature_2 m_K"] = virtual_temperature(
        derived_data["potential temperature_2 m_K"], derived_data["mixing ratio_2 m_g kg-1"])
    derived_data["air density_10 m_kg m-3"] = air_density(virtual_temperature(derived_data["temperature_10 m_K"],
                                                                              derived_data["mixing ratio_10 m_g kg-1"]),
                                                          derived_data["pressure_2 m_hPa"])
    for height in [10, 20, 40]:
        derived_data["wind speed_{0:d} m_m s-1".format(height)] = combined_data[("tower", "F_{0:d}m".format(height))]
        derived_data["wind direction_{0:d} m_degrees".format(height)] = combined_data[("tower", "D_{0:d}m".format(height))]
        derived_data["u wind_{0:d} m_m s-1".format(height)], derived_data["v wind_{0:d} m_m s-1".format(height)] = \
            wind_components(derived_data["wind speed_{0:d} m_m s-1".format(height)],
                            derived_data["wind direction_{0:d} m_degrees".format(height)])
    derived_data["soil temperature_0 cm_K"] = celsius_to_kelvin(combined_data[("soil", "TS00")])
    derived_data["soil temperature_4 cm_K"] = celsius_to_kelvin(combined_data[("soil", "TS04")])
    derived_data["soil potential temperature_0 cm_K"] = potential_temperature(derived_data["soil temperature_0 cm_K"],
                                                                             derived_data["pressure_2 m_hPa"])
    derived_data["soil potential temperature_4 cm_K"] = potential_temperature(derived_data["soil temperature_4 cm_K"],
                                                                             derived_data["pressure_2 m_hPa"])
    derived_data["soil water content_3 cm_m3 m-3"] = combined_data[("soil_water", "TH03")]
    #derived_data["soil water content_8 cm_m3 m-3"] = combined_data[("soil_water", "TH08")]
    derived_data["moisture availability_3 cm_"] = moisture_availability(derived_data["soil water content_3 cm_m3 m-3"])
    #derived_data["moisture availability_8 cm_"] = moisture_availability(derived_data["soil water content_8 cm_m3 m-3"])
    derived_data["upward longwave irradiance_0 m_W m-2"] = combined_data[("irrad", "LWU")]
    derived_data["downward longwave irradiance_0 m_W m-2"] = combined_data[("irrad", "LWD")]
    derived_data["upward shortwave irradiance_0 m_W m-2"] = combined_data[("irrad", "SWU")]

    derived_data["skin temperature_0 m_K"] = skin_temperature(derived_data["upward longwave irradiance_0 m_W m-2"])
    derived_data["skin potential temperature_0 m_K"] = potential_temperature(derived_data["skin temperature_0 m_K"],
                                                                             derived_data["pressure_2 m_hPa"])
    derived_data["skin saturation mixing ratio_0 m_g kg-1"] = saturation_mixing_ratio(derived_data["skin temperature_0 m_K"],
                                                                                      derived_data["pressure_2 m_hPa"])
    derived_data["skin virtual potential temperature_0 m_K"] = virtual_temperature(derived_data["skin potential temperature_0 m_K"],
                                                                                   derived_data["skin saturation mixing ratio_0 m_g kg-1"])
    for height in [10, 20, 40]:
        derived_data[f"potential temperature skin change_{height:d} m_K m-1"] = \
            (derived_data["skin potential temperature_0 m_K"] - derived_data[f"potential temperature_{height:d} m_K"]) \
            / height
        derived_data[f"virtual potential temperature skin change_{height:d} m_K m-1"] = \
            (derived_data["skin virtual potential temperature_0 m_K"] - derived_data[f"virtual potential temperature_{height:d} m_K"]) \
            / height
        derived_data[f"mixing ratio skin change_{height:d} m_g kg-1 m-1"] = \
            (derived_data["skin saturation mixing ratio_0 m_g kg-1"] - derived_data[f"mixing ratio_{height:d} m_g kg-1"]) \
            / height
    derived_data["friction velocity_surface_m s-1"] = np.maximum(combined_data[("flux", "UST")], 0.001)
    derived_data["sensible heat flux_surface_W m-2"] = combined_data[("flux", "H")]
    derived_data["latent heat flux_surface_W m-2"] = combined_data[("flux", "LE")]
    derived_data["soil heat flux_surface_W m-2"] = combined_data[("flux", "G0")]
    if reflect_counter_gradient:
        sh_counter_gradient = (derived_data[f"potential temperature skin change_10 m_K m-1"] *
                               derived_data["sensible heat flux_surface_W m-2"]) < 0
        lh_counter_gradient = (derived_data[f"mixing ratio skin change_10 m_g kg-1 m-1"] *
                               derived_data["latent heat flux_surface_W m-2"]) < 0
        derived_data.loc[sh_counter_gradient,
                         "sensible heat flux_surface_W m-2"] = -derived_data.loc[sh_counter_gradient,
                                                                                 "sensible heat flux_surface_W m-2"]
        derived_data.loc[lh_counter_gradient,
                         "latent heat flux_surface_W m-2"] = -derived_data.loc[lh_counter_gradient,
                                                                               "latent heat flux_surface_W m-2"]
    derived_data["temperature scale_surface_K"] = temperature_scale(derived_data["sensible heat flux_surface_W m-2"],
                                                                    derived_data["air density_10 m_kg m-3"],
                                                                    derived_data["friction velocity_surface_m s-1"])
    derived_data["moisture scale_surface_g kg-1"] = moisture_scale(derived_data["latent heat flux_surface_W m-2"],
                                                                   derived_data["air density_10 m_kg m-3"],
                                                                   derived_data["friction velocity_surface_m s-1"])
    derived_data["bulk richardson_10 m_"] = bulk_richardson_number(derived_data["potential temperature_10 m_K"],
                                                                   10,
                                                                   derived_data["mixing ratio_10 m_g kg-1"],
                                                                   derived_data["skin virtual potential temperature_0 m_K"],
                                                                   derived_data["wind speed_10 m_m s-1"])
    derived_data["bulk richardson_2 m_"] = bulk_richardson_number(derived_data["potential temperature_2 m_K"],
                                                                  2,
                                                                  derived_data["mixing ratio_2 m_g kg-1"],
                                                                  derived_data["skin virtual potential temperature_0 m_K"],
                                                                  derived_data["wind speed_10 m_m s-1"])
    derived_data["bulk richardson_10-2 m_"] = bulk_richardson_number(derived_data["potential temperature_10 m_K"],
                                                                     10,
                                                                     derived_data["mixing ratio_10 m_g kg-1"],
                                                                     derived_data["virtual potential temperature_2 m_K"],
                                                                     derived_data["wind speed_10 m_m s-1"])
    derived_data["obukhov length_surface_m"] = obukhov_length(derived_data["potential temperature_10 m_K"],
                                                              derived_data["temperature scale_surface_K"],
                                                              derived_data["friction velocity_surface_m s-1"])
    if average_period is not None:
        derived_data = derived_data.rolling(window=average_period).mean()
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


def filter_counter_gradient_data(data, gradient_column="potential temperature skin change_10 m_K m-1",
                                 flux_column="sensible heat flux_surface_W m-2"):
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


def process_idaho_data(csv_path, out_file, idaho_lat=43.5897, idaho_lon=-112.9399,
                        elevation=1492.6, reflect_counter_gradient=False, average_period=None):
    """
    Site data at https://www.noaa.inel.gov/projects/INLMet/MesonetLocations.pdf

    Args:
        csv_path:
        out_file:
        nan_column:
        idaho_lat:
        idaho_lon:
        elevation:
        reflect_counter_gradient:
        average_period:

    Returns:

    """
    idaho_flux_data = pd.read_csv(join(csv_path, "FRD_ECFlux_2015-2017.csv"), na_values=[-9999.0])
    idaho_met_data = pd.read_csv(join(csv_path, "FRD_TallTower_Met_2015-2017.csv"), na_values=[-9999.0])
    idaho_rad_data = pd.concat([pd.read_csv(join(csv_path, f"FRD_FLXStation_radiation_{year:d}.csv"),
                                            na_values=[-999]) for year in range(2015, 2018)])
    idaho_flux_no_missing = idaho_flux_data.dropna(axis=0, how="any")
    idaho_met_no_missing = idaho_met_data.dropna(axis=0, how="any")
    idaho_rad_no_missing = idaho_rad_data.dropna(axis=0, how="any")
    idaho_met_no_missing.index = pd.DatetimeIndex(idaho_met_no_missing[['Year', 'Month', 'Day', 'Hour', 'Minute']].apply(
        lambda s: datetime.datetime(*s), axis=1))
    flux_dt = idaho_flux_no_missing.date.apply(
        lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))
    flux_dt = flux_dt + ' ' + idaho_flux_no_missing['time']
    idaho_flux_no_missing.index = pd.DatetimeIndex(flux_dt)
    idaho_rad_no_missing.index = pd.DatetimeIndex(idaho_rad_no_missing[['Year', 'Month', 'Day', 'Hour', 'Minute']].apply(
        lambda s: datetime.datetime(*s), axis=1))
    idaho_met_no_missing.drop(["Year", "Month", "Day", "Hour", "Minute"], axis=1, inplace=True)
    idaho_rad_no_missing.drop(["Year", "Month", "Day", "Hour", "Minute"], axis=1, inplace=True)
    idaho_flux_no_missing.drop(["date", "time", "DOY", "daytime"], axis=1, inplace=True)
    result = pd.concat([idaho_met_no_missing, idaho_flux_no_missing, idaho_rad_no_missing], axis=1, join="inner")
    print(result)
    for col in result.columns:
        print(col)
    solpos = get_solarposition(result.index, idaho_lat, idaho_lon, altitude=elevation)

    result['solar_zenith_angle'] = np.array(solpos['zenith'])

    # Need to convert Idaho data from celsius to kelvin
    result['2m Temp K'] = celsius_to_kelvin(result['2m Temp C'])
    result['10m Temp K'] = celsius_to_kelvin(result['10m Temp C'])
    result['15m Temp K'] = celsius_to_kelvin(result['15m Temp C'])
    result['45m Temp K'] = celsius_to_kelvin(result['45m Temp C'])

    # Need to convert Idaho data to wind components from speed and direction
    result['2m U-Wind m/s'], result['2m V-Wind m/s'] = wind_components(result['2m Wind Speed m/s'],
                                                                       result['2m Wind Dir deg'])
    result['10m U-Wind m/s'], result['10m V-Wind m/s'] = wind_components(result['10m Wind Speed m/s'],
                                                                         result['10m Wind Dir deg'])
    result['15m U-Wind m/s'], result['15m V-Wind m/s'] = wind_components(result['15m Wind Speed m/s'],
                                                                         result['15m Wind Dir deg'])
    result['45m U-Wind m/s'], result['45m V-Wind m/s'] = wind_components(result['45m Wind Speed m/s'],
                                                                         result['45m Wind Dir deg'])

    # Need to convert temperatures in inches of mercury to hectopascals
    print(inHg_to_hpa(result['BP inches Hg'].values).shape)
    result.loc[:, 'pressure hpa'] = inHg_to_hpa(result['BP inches Hg'].values)

    # Need to convert from Kelvin to Potential Temperature
    result['2m potential temperature K'] = potential_temperature(result['2m Temp K'], result['pressure hpa'])
    result['10m potential temperature K'] = potential_temperature(result['10m Temp K'],
                                                                  result['pressure hpa'])
    result['15m potential temperature K'] = potential_temperature(result['15m Temp K'],
                                                                  result['pressure hpa'])
    result['45m potential temperature K'] = potential_temperature(result['45m Temp K'],
                                                                  result['pressure hpa'])

    # Compute the friction velocity
    result['friction velocity m_s'] = friction_velocity(result['Tau'], result['air_density'])

    # Compute the temperature scale
    result['temperature scale K'] = temperature_scale(result['H'], result['air_density'], result['friction velocity m_s'])

    # Compute the moisture scale
    result['moisture scale g_kg'] = moisture_scale(result['LE'], result['air_density'], result['friction velocity m_s'])

    # Compute the Monin Obukhov Length
    result['obukhov length m'] = obukhov_length(result['2m potential temperature K'], result['temperature scale K'],
                                              result['friction velocity m_s'], von_karman_constant=0.4)

    # Compute the mixing ratio
    result['2m mixing ratio g_kg'] = mixing_ratio(result['2m Temp C'],
                                                  result['2m RH %'], result["pressure hpa"])

    # Compute the virtual potential temperature
    result['2m virtual potential temperature K'] = virtual_temperature(result['2m potential temperature K'],
                                                                     result['2m mixing ratio g_kg'])
    # Compute the Bulk Richardson Number
    result['bulk richardson number'] = bulk_richardson_number(result['2m potential temperature K'], 2,
                                                              result['2m mixing ratio g_kg'],
                                                              result['2m virtual potential temperature K'],
                                                              result['2m Wind Speed m/s'])
    # Try computing Bulk Richardson Number between 10M and 2M
    # Compute the Bulk Richardson Number
    result['10m bulk richardson number'] = bulk_richardson_number(result['10m potential temperature K'], 10,
                                                                  result['2m mixing ratio g_kg'],
                                                                  result['2m virtual potential temperature K'],
                                                                  result['2m Wind Speed m/s'])
    # Try computing Bulk Richardson Number between 2M and 5cm soil temperature
    # Compute the Bulk Richardson Number
    # Compute the virtual potential temperature

    result['5cm Soil Temp K'] = celsius_to_kelvin(result['5cm Soil Temp C'])
    result['5cm potential temperature K'] = potential_temperature(result['5cm Soil Temp K'],
                                                                  result['pressure hpa'])
    result['5cm virtual potential temperature'] = virtual_temperature(result['5cm potential temperature K'],
                                                                      result['2m mixing ratio g_kg'])
    result['Soil bulk richardson number'] = bulk_richardson_number(result['2m potential temperature K'], 10,
                                                                   result['2m mixing ratio g_kg'],
                                                                   result['5cm virtual potential temperature'],
                                                                   result['2m Wind Speed m/s'])

    result["roughness length surface m"] = np.minimum(9, np.maximum(0.001, 10 * np.exp(
        -0.4 * np.maximum(result["10m Wind Speed m/s"], 1) / result["friction velocity m_s"])))
    result['10m bulk richardson number'].drop(result['10m bulk richardson number'].idxmax())
    result["BRN_sign"] = np.sign(result['10m bulk richardson number'])
    result['Soil bulk richardson number'].drop(result['Soil bulk richardson number'].idxmin())
    result["BRN_sign_5cm"] = np.sign(result['Soil bulk richardson number'])
    result["skin temperature K"] = skin_temperature(result["Lwout (W/m^2)"])
    result["skin potential temperature K"] = potential_temperature(result["skin temperature K"], result["pressure hpa"])
    result["skin saturation mixing ratio g_kg"] = saturation_mixing_ratio(result["skin temperature K"],
                                                                          result["pressure hpa"])
    result['skin bulk richardson number'] = bulk_richardson_number(result['10m potential temperature K'], 10,
                                                                  result['2m mixing ratio g_kg'],
                                                                  result['skin potential temperature K'],
                                                                  result['2m Wind Speed m/s'])
    if average_period is not None:
        result = result.rolling(window=average_period).mean()
        result = result.dropna()
    result.to_csv(out_file, index_label="Time")
    return