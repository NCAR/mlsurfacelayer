import pandas as pd
from glob import glob
from os.path import join
from .derived import *


def process_cabauw_data(csv_path, out_file, nan_column=("soil", "TS00")):
    """
    This function loads all of the cabauw data files and then calculates the relevant derived quantities necessary
    to build the machine learning parameterization.

    Args:
        csv_path: Path to all csv files.

    Returns:
        `pandas.DataFrame` containing derived data.
    """
    csv_files = sorted(glob(join(csv_path, "*.csv")))
    file_types = [csv_file.split("/")[-1].split("_")[1] for csv_file in csv_files]
    data = dict()
    for c, csv_file in enumerate(csv_files):
        print(csv_file)
        data[file_types[c]] = pd.read_csv(csv_file, na_values=[-9999.0])
        data[file_types[c]].index = pd.to_datetime(data[file_types[c]]["TimeStr"], format="%Y%m%d.%H:%M")
    print("combine data")
    combined_data = pd.concat(data, axis=1, join="inner")
    combined_data = combined_data.loc[~pd.isna(combined_data[nan_column])]
    derived_columns = ["temperature_10 m_K",
                       "pressure_2 m_hPa",
                       "potential temperature_10 m_K",
                       "mixing ratio_10 m_g kg-1",
                       "virtual potential temperature_10 m_K",
                       "air density_10 m_kg m-3",
                       "wind speed_10 m_m s-1",
                       "wind direction_10 m_m s-1",
                       "u wind_10 m_m s-1",
                       "v wind_10 m_m s-1",
                       "mixing ratio_2 m_g kg-1",
                       "temperature_0 m_K",
                       "potential temperature_0 m_K",
                       "virtual potential temperature_0 m_K",
                       "friction velocity_surface_K",
                       "temperature scale_surface_K",
                       "moisture scale_surface_g kg-1",
                       "bulk richardson_surface_",
                       "obukhov length_surface_m"
                       ]
    derived_data = pd.DataFrame(index=combined_data.index, columns=derived_columns, dtype=float)
    print("calculate derived variables")
    derived_data["temperature_10 m_K"] = combined_data[("tower", "TA_10m")]
    derived_data["pressure_2 m_hPa"] = combined_data[("surface", "P0")]
    derived_data["potential temperature_10 m_K"] = potential_temperature(derived_data["temperature_10 m_K"],
                                                                         derived_data["pressure_2 m_hPa"])
    derived_data["mixing ratio_10 m_g kg-1"] = combined_data[("tower", "Q_10m")]
    derived_data["virtual potential temperature_10 m_K"] = virtual_temperature(
        derived_data["potential temperature_10 m_K"], derived_data["mixing ratio_10 m_g kg-1"])
    derived_data["air density_10 m_kg m-3"] = air_density(virtual_temperature(derived_data["temperature_10 m_K"],
                                                                              derived_data["mixing ratio_10 m_g kg-1"]),
                                                          derived_data["pressure_2 m_hPa"])
    derived_data["wind speed_10 m_m s-1"] = combined_data[("tower", "F_10m")]
    derived_data["wind direction_10 m_m s-1"] = combined_data[("tower", "D_10m")]
    derived_data["u wind_10 m_m s-1"], \
    derived_data["v wind_10 m_m s-1"] = wind_components(derived_data["wind speed_10 m_m s-1"],
                                                        derived_data["wind direction_10 m_m s-1"])
    derived_data["mixing ratio_2 m_g kg-1"] = combined_data[("tower", "Q_2m")]
    derived_data["temperature_0 m_K"] = celsius_to_kelvin(combined_data[("soil", "TS00")])
    derived_data["potential temperature_0 m_K"] = potential_temperature(derived_data["temperature_0 m_K"],
                                                                        derived_data["pressure_2 m_hPa"])
    derived_data["virtual potential temperature_0 m_K"] = virtual_temperature(derived_data["potential temperature_0 m_K"],
                                                                              derived_data["mixing ratio_2 m_g kg-1"])
    derived_data["friction velocity_surface_K"] = combined_data[("flux", "UST")]
    derived_data["temperature scale_surface_K"] = temperature_scale(combined_data[("flux", "H")],
                                                                    derived_data["air density_10 m_kg m-3"],
                                                                    derived_data["friction velocity_surface_K"])
    derived_data["moisture scale_surface_g kg-1"] = moisture_scale(combined_data[("flux", "LE")],
                                                                   derived_data["air density_10 m_kg m-3"],
                                                                   derived_data["friction velocity_surface_K"])
    derived_data["bulk richardson_surface_"] = bulk_richardson_number(derived_data["potential temperature_10 m_K"],
                                                                      10,
                                                                      derived_data["mixing ratio_10 m_g kg-1"],
                                                                      derived_data["virtual potential temperature_0 m_K"],
                                                                      derived_data["wind speed_10 m_m s-1"])
    derived_data["obukhov length_surface_m"] = obukhov_length(derived_data["potential temperature_10 m_K"],
                                                              derived_data["temperature scale_surface_K"],
                                                              derived_data["friction velocity_surface_K"])
    derived_data.to_csv(out_file, index_label="Time")
    return derived_data

