

def potential_temperature(temperature_k, pressure_hpa, pressure_reference_hpa=1000.0):
    """
    Convert temperature to potential temperature based on the available pressure. Potential temperature is at a
    reference pressure of 1000 mb.

    Args:
        temperature_k: The air temperature in units K
        pressure_hpa: The atmospheric pressure in units hPa
        pressure_reference_hpa: The reference atmospheric pressure for the potential temperature in hPa;
            default 1000 hPa

    Returns:
        The potential temperature in units K
    """
    return temperature_k * (pressure_reference_hpa / pressure_hpa) ** (2.0 / 7.0)


def virtual_temperature(temperature_k, mixing_ratio_kg_kg):
    """
    Convert temperature and mixing ratio to virtual temperature.

    Args:
        temperature_k: The temperature or potential temperature in units K.
        mixing_ratio_kg_kg: The mixing ratio in units kg kg-1.

    Returns:
        The virtual temperature in units K.
    """
    return temperature_k * (1 + 0.61 * mixing_ratio_kg_kg)


def air_density(virtual_temperature_k, pressure_hPa):
    """
    Calculate the density of air based on the ideal gas law, virtual temperature, and pressure.

    Args:
        virtual_temperature_k: The virtual temperature in units K.
        pressure_hPa: The pressure in units hPa.

    Returns:
        The density of air in units kg m-3.
    """
    gas_constant = 287.0
    return pressure_hPa * 100.0 / (gas_constant * virtual_temperature_k)


def temperature_scale(sensible_heat_flux_W_m2, air_density_kg_m3, friction_velocity_m_s):
    """
    Caclulate the temperature turbulence scale value theta* from the sensible heat flux.

    Args:
        sensible_heat_flux_W_m2: The sensible heat flux in units W m-2.
        air_density_kg_m3: The density of air in units kg m-3.
        friction_velocity_m_s: The friction velocity in units m s-1.

    Returns:
        The temperature turbulence scale value in units K.
    """
    return -sensible_heat_flux_W_m2 / (air_density_kg_m3 * 287.0 * 7.0 / 2.0 * friction_velocity_m_s)


def moisture_scale(latent_heat_flux_W_m2, air_density_kg_m3, friction_velocity_m_s):
    """
    Calculate the turblulent moisture scale factor from the latent heat flux.

    Args:
        latent_heat_flux_W_m2: Latent heat flux in units W m-2
        air_density_kg_m3: Density of air in units kg m-3
        friction_velocity_m_s: The friction velocity (u*) in units m s-1

    Returns:
        The turbulent moisture scale factor in kg kg-1
    """
    latent_heat_of_vaporization_J_kg = 2264705.0  # J kg-1
    return latent_heat_flux_W_m2 / (latent_heat_of_vaporization_J_kg * air_density_kg_m3 * friction_velocity_m_s)


def bulk_richardson_number(potential_temperature_k, height,
                           mixing_ratio_kg_kg, virtual_potential_skin_temperature_k, wind_speed_m_s):
    """
    Calculate the bulk Richardson number, a measure of stability.

    Args:
        potential_temperature_k: The potential or virtual potential temperature in K
        height: The height at which the potential temperature calculation is performed in m.
        mixing_ratio_kg_kg: The mixing ratio at the same height as the potential temperature in units kg kg-1.
        virtual_potential_skin_temperature_k: The virtual potential temperature at the surface
        wind_speed_m_s: The wind speed in m s-1

    Returns:

    """
    g = 9.81  # m s-2
    virtual_potential_temperature_k = virtual_temperature(potential_temperature_k, mixing_ratio_kg_kg)
    return g / potential_temperature_k * height * (virtual_potential_temperature_k
                                                   - virtual_potential_skin_temperature_k) / wind_speed_m_s ** 2


def obukhov_length(potential_temperature_k, temperature_scale_k, friction_velocity_m_s, von_karman_constant=0.4):
    """
    Caclulates the Obukhov length, a measure of stability based on the friction velocity and temperature scale.

    Args:
        potential_temperature_k: The potential temperature in units K
        temperature_scale_k: The turbulent temperature scale (theta*) in units K
        friction_velocity_m_s: The friction velocity (u*) in units m s-1
        von_karman_constant: The von Karman constant (default=0.4)

    Returns:
        The Obukhov length in units m.
    """
    g = 9.81 # m s-2
    return friction_velocity_m_s ** 2 * potential_temperature_k / (von_karman_constant * g * temperature_scale_k)

