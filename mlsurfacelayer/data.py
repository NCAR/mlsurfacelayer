import pandas as pd
from glob import glob
from os.path import join
from .derived import *
from pvlib.solarposition import get_solarposition


def process_cabauw_data(csv_path, out_file, nan_column=("soil_water", "TH03")):
    """
    This function loads all of the cabauw data files and then calculates the relevant derived quantities necessary
    to build the machine learning parameterization.

    Args:
        csv_path: Path to all csv files.
        out_file: Where derived data are written to.
        nan_column: Column used to filter bad examples.

    Returns:
        `pandas.DataFrame` containing derived data.
    """
    cabauw_lat = 51.971
    cabauw_lon = 4.926
    elevation = -0.7
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
    derived_columns = ["global horizontal irradiance_0 m_W m-2",
                       "zenith_0 m_degrees",
                       "azimuth_0 m_degrees",
                       "temperature_10 m_K",
                       "temperature_2 m_K",
                       "pressure_2 m_hPa",
                       "potential temperature_10 m_K",
                       "potential temperature_2 m_K",
                       "mixing ratio_10 m_g kg-1",
                       "relative humidity_10 m_%",
                       "virtual potential temperature_10 m_K",
                       "air density_10 m_kg m-3",
                       "wind speed_10 m_m s-1",
                       "wind direction_10 m_m s-1",
                       "wind speed_20 m_m s-1",
                       "wind direction_20 m_m s-1",
                       "wind speed_40 m_m s-1",
                       "wind direction_40 m_m s-1",
                       "u wind_10 m_m s-1",
                       "v wind_10 m_m s-1",
                       "u wind_20 m_m s-1",
                       "v wind_20 m_m s-1",
                       "u wind_40 m_m s-1",
                       "v wind_40 m_m s-1",
                       "mixing ratio_2 m_g kg-1",
                       "virtual potential temperature_10 m_K",
                       "relative humidity_2 m_%",
                       "soil temperature_0 cm_K",
                       "soil temperature_4 cm_K",
                       "soil temperature_6 cm_K",
                       "soil potential temperature_0 cm_K",
                       "soil potential temperature_4 cm_K",
                       "soil potential temperature_6 cm_K",
                       "soil water content_3 cm_m3 m-3",
                       "soil water content_8 cm_m3 m-3",
                       "friction velocity_surface_K",
                       "temperature scale_surface_K",
                       "moisture scale_surface_g kg-1",
                       "bulk richardson_surface_",
                       "obukhov length_surface_m"
                       ]
    derived_data = pd.DataFrame(index=combined_data.index, columns=derived_columns, dtype=float)
    solar_data = get_solarposition(combined_data.index, cabauw_lat, cabauw_lon, altitude=elevation, method="nrel_numba")
    print("calculate derived variables")
    derived_data["global horizontal irradiance_0 m_W m-2"] = combined_data[("surface", "SWD")]
    derived_data["zenith_0 m_degrees"] = solar_data["zenith"]
    derived_data["azimuth_0 m_degrees"] = solar_data["azimuth"]
    derived_data["temperature_10 m_K"] = combined_data[("tower", "TA_10m")]
    derived_data["temperature_2 m_K"] = combined_data[("tower", "TA_2m")]
    derived_data["pressure_2 m_hPa"] = combined_data[("surface", "P0")]
    derived_data["potential temperature_10 m_K"] = potential_temperature(derived_data["temperature_10 m_K"],
                                                                         derived_data["pressure_2 m_hPa"])
    derived_data["potential temperature_2 m_K"] = potential_temperature(derived_data["temperature_2 m_K"],
                                                                         derived_data["pressure_2 m_hPa"])
    derived_data["mixing ratio_10 m_g kg-1"] = combined_data[("tower", "Q_10m")]
    derived_data["relative humidity_10 m_%"] = combined_data[("tower", "RH_10m")]
    derived_data["relative humidity_2 m_%"] = combined_data[("tower", "RH_2m")]
    derived_data["virtual potential temperature_10 m_K"] = virtual_temperature(
        derived_data["potential temperature_10 m_K"], derived_data["mixing ratio_10 m_g kg-1"])
    derived_data["virtual potential temperature_2 m_K"] = virtual_temperature(
        derived_data["potential temperature_2 m_K"], derived_data["mixing ratio_2 m_g kg-1"])
    derived_data["air density_10 m_kg m-3"] = air_density(virtual_temperature(derived_data["temperature_10 m_K"],
                                                                              derived_data["mixing ratio_10 m_g kg-1"]),
                                                          derived_data["pressure_2 m_hPa"])
    for height in [10, 20, 40]:
        derived_data["wind speed_{0:d} m_m s-1".format(height)] = combined_data[("tower", "F_{0:d}m".format(height))]
        derived_data["wind direction_{0:d} m_m s-1".format(height)] = combined_data[("tower", "D_{0:d}m".format(height))]
        derived_data["u wind_{0:d} m_m s-1".format(height)], derived_data["v wind_{0:d} m_m s-1".format(height)] = \
            wind_components(derived_data["wind speed_{0:d} m_m s-1".format(height)],
                            derived_data["wind direction_{0:d} m_m s-1".format(height)])
    derived_data["mixing ratio_2 m_g kg-1"] = combined_data[("tower", "Q_2m")]
    derived_data["soil temperature_0 cm_K"] = celsius_to_kelvin(combined_data[("soil", "TS00")])
    derived_data["soil temperature_4 cm_K"] = celsius_to_kelvin(combined_data[("soil", "TS04")])
    derived_data["soil temperature_6 cm_K"] = celsius_to_kelvin(combined_data[("soil", "TS06")])
    derived_data["soil potential temperature_0 m_K"] = potential_temperature(derived_data["soil temperature_0 cm_K"],
                                                                             derived_data["pressure_2 m_hPa"])
    derived_data["soil potential temperature_0 m_K"] = potential_temperature(derived_data["soil temperature_4 cm_K"],
                                                                             derived_data["pressure_2 m_hPa"])
    derived_data["soil potential temperature_0 m_K"] = potential_temperature(derived_data["soil temperature_6 cm_K"],
                                                                             derived_data["pressure_2 m_hPa"])
    derived_data["soil water content_3 cm_m3 m-3"] = combined_data[("soil_water", "TH03")]
    derived_data["soil water content_8 cm_m3 m-3"] = combined_data[("soil_water", "TH08")]
    derived_data["friction velocity_surface_m s-1"] = combined_data[("flux", "UST")]
    derived_data["temperature scale_surface_K"] = temperature_scale(combined_data[("flux", "H")],
                                                                    derived_data["air density_10 m_kg m-3"],
                                                                    derived_data["friction velocity_surface_K"])
    derived_data["moisture scale_surface_g kg-1"] = moisture_scale(combined_data[("flux", "LE")],
                                                                   derived_data["air density_10 m_kg m-3"],
                                                                   derived_data["friction velocity_surface_K"])
    derived_data["bulk richardson_10 m_"] = bulk_richardson_number(derived_data["potential temperature_10 m_K"],
                                                                   10,
                                                                   derived_data["mixing ratio_10 m_g kg-1"],
                                                                   derived_data["soil potential temperature_4 cm_K"],
                                                                   derived_data["wind speed_10 m_m s-1"])
    derived_data["bulk richardson_2 m_"] = bulk_richardson_number(derived_data["potential temperature_2 m_K"],
                                                                  2,
                                                                  derived_data["mixing ratio_2 m_g kg-1"],
                                                                  derived_data["soil potential temperature_4 cm_K"],
                                                                  derived_data["wind speed_10 m_m s-1"])
    derived_data["obukhov length_surface_m"] = obukhov_length(derived_data["potential temperature_10 m_K"],
                                                              derived_data["temperature scale_surface_K"],
                                                              derived_data["friction velocity_surface_K"])
    derived_data.to_csv(out_file, index_label="Time")
    return derived_data

