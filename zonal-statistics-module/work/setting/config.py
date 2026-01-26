from pydantic_settings import BaseSettings
from typing import Dict, List

class Settings(BaseSettings):

    chrisp_bands_list: list = ['precipitation']
    
    era5_bands_list: list = ['dewpoint_temperature_2m',
                            'dewpoint_temperature_2m_min',
                            'dewpoint_temperature_2m_max',
                            'temperature_2m',
                            'temperature_2m_min',
                            'temperature_2m_max',
                            'skin_temperature', 
                            'soil_temperature_level_1', 
                            'soil_temperature_level_2', 
                            'soil_temperature_level_3', 
                            'soil_temperature_level_4',
                            'lake_bottom_temperature',
                            'lake_ice_depth',
                            'lake_ice_temperature',
                            'lake_mix_layer_depth',
                            'lake_mix_layer_temperature', 
                            'lake_shape_factor', 
                            'lake_total_layer_temperature',
                            'snow_albedo',
                            'snow_cover',
                            'snow_density',
                            'snow_depth',
                            'snow_depth_water_equivalent',
                            'snowfall_sum',
                            'snowmelt_sum',
                            'temperature_of_snow_layer',
                            'skin_reservoir_content',
                            'volumetric_soil_water_layer_1',
                            'volumetric_soil_water_layer_2',
                            'volumetric_soil_water_layer_3',
                            'volumetric_soil_water_layer_4',
                            'forecast_albedo',
                            'surface_latent_heat_flux_sum',
                            'surface_net_solar_radiation_sum',
                            'surface_net_thermal_radiation_sum',
                            'surface_sensible_heat_flux_sum',
                            'surface_solar_radiation_downwards_sum',
                            'surface_thermal_radiation_downwards_sum',
                            'evaporation_from_bare_soil_sum',
                            'evaporation_from_open_water_surfaces_excluding_oceans_sum',
                            'evaporation_from_the_top_of_canopy_sum',
                            'evaporation_from_vegetation_transpiration_sum',
                            'potential_evaporation_sum',
                            'runoff_sum',
                            'snow_evaporation_sum',
                            'sub_surface_runoff_sum',
                            'surface_runoff_sum',
                            'total_evaporation_sum',
                            'u_component_of_wind_10m',
                            'v_component_of_wind_10m',
                            'surface_pressure',
                            'total_precipitation_sum',
                            'leaf_area_index_high_vegetation',
                            'leaf_area_index_low_vegetation']

    modis_ndvi_evi_bands_list: list = ['NDVI', 'EVI']
    modis_lst_bands_list: list = ['LST_Day','LST_Night','LST_Mean']
    modis_nadir_bands_list: list = ['NDVI','EVI','SAVI','NDWI_Gao', 'NDWI_Mc', 'MNDWI']

    bands_introduction: str = 'Environmental Parameters'
    shp_name: str = 'shp file'
    regional_category: str = 'Location_ID'
    statistic_name: str = 'Statistics'
    statistic_list: list = ['MEAN','MAXIMUM', 'MINIMUM', 'MEDIAN', 'STD', 'VARIANCE', 'SUM']
    statics_world_list: list = ['SUM', 'PERCENTAGE']

    # account cinfig
    service_account: str = 'fyakghoon226677@ee-hoolu.iam.gserviceaccount.com'
    service_json: str = 'ee-hoolu-0ef4b688458f.json'
    crs: str = 'EPSG:4326'

settings = Settings()