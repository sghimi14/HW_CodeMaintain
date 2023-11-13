#IMPORTING ALL THE LIBRARIES AND MODULES
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import os
import seaborn as sns
import glob
from shapely.geometry import Poin
from sklearn.metrics import mean_squared_error, mean_absolute_error


#IMPORTING SHAPEFILES OF CRB, UPPER BASIN,LOWER BASIN AND CHANGING PROJECTION
#Change the path for reading shapefiles
crb_shp = gpd.read_file('/Users/sghimi14/Documents/Research/Analysis_Basin_Shapefiles/basin_CRB_poly.shp')
crb = crb_shp.to_crs('epsg:4326')

lower_shp = gpd.read_file('/Users/sghimi14/Documents/Research/Analysis_Basin_Shapefiles/basin_LowerBasin_poly.shp')
lower = lower_shp.to_crs('epsg:4326')

upper_shp = gpd.read_file('/Users/sghimi14/Documents/Research/Analysis_Basin_Shapefiles/basin_UpperBasin_poly.shp')
upper = upper_shp.to_crs('epsg:4326')


#LOCATING AND COMBINING ALL NETCDF FILES
#Change the path for reading netcdf files
file_paths_2018_2023 = glob.glob('Outputs_2018_2023/*.nc')
file_paths_2010_2017 = glob.glob('Outputs_2010_2017/*.nc')
file_paths_2004_2009 = glob.glob('Outputs_2004_2009/*.nc')

all_file_paths = file_paths_2018_2023 + file_paths_2010_2017 + file_paths_2004_2009


#ANALYSIS
snodas_data = xr.open_mfdataset(all_file_paths)

SWE = snodas_data.SWE
SWE.rio.write_crs('epsg:4326', inplace=True)

#If you want to change the analysis period, change the start and end date
start_date = '2004-01-01'
end_date = '2023-08-06'
new_time_values = pd.date_range(start=start_date, end=end_date, freq='D')
snodas_dates = SWE['Time'].astype(str).values.tolist()
snodas_dates = pd.to_datetime(snodas_dates)
missing_dates = set(new_time_values) - set(snodas_dates)
new_time_values = new_time_values[~new_time_values.isin(missing_dates)]
snodas_clipped = SWE.rio.clip(crb.geometry)
snodas_clipped['Time'] = new_time_values


#METADATA AND SITE EXTRACTION
metadata = pd.read_csv('snotel_metadata.csv')
snotel_data = pd.read_csv('snotel_pivoted.csv', index_col=0, parse_dates=True)

sites = snotel_data.columns.tolist()
sites = sorted([int(i) for i in sites])
req_sites = metadata[metadata['site_id'].isin(sites)]

req_sites = req_sites[['site_id', 'latitude', 'longitude', 'elev']]
req_sites = req_sites.set_index('site_id').sort_index()
sites_dict = req_sites.to_dict(orient='index')

for each_st in sites_dict:
    station_name = each_st
    station_info = sites_dict[each_st]

    lat = station_info['latitude']
    lon = station_info['longitude']
    clip_point = Point(lon, lat)

    SWE_pixel = snodas_clipped.rio.clip([clip_point], crs='epsg:4326', all_touched=False, from_disk=True) #All touched equals to False will clip only the pixel whose centroid falls inside the shapefile boundary
    SWE_df = SWE_pixel.to_dataframe()
    SWE_df = SWE_df.reset_index(level=['y', 'x'], drop=True)
    SWE_df = SWE_df.drop(columns='spatial_ref')

    SWE_df.to_csv(f'SNODAS_ts/{station_name}.csv')

snotel_data = pd.read_csv('snotel_pivoted.csv', index_col=0, parse_dates=True)


#PLOT CREATION
files = os.listdir('SNODAS_ts')
files = [file for file in files if file.endswith('.csv')

elevation_values = []
corr_values = []
rmse_values = []
mae_values = []
bias_values = []

for file in files:
    name = file.split('.')[0]
    snodas = pd.read_csv(f'SNODAS_ts/{file}', index_col=0, parse_dates=True)
    snodas = snodas.rename(columns={'SWE': 'SNODAS'})
    snotel = snotel_data[name].to_frame()
    snotel = snotel.rename(columns={name: 'SNOTEL'})

    merged_data = snotel.join(snodas, how='inner')
    merged_data = merged_data.resample('M').mean()
    merged_data = merged_data.dropna()

    corr = merged_data['SNODAS'].corr(merged_data['SNOTEL'])
    rmse = np.sqrt(mean_squared_error(merged_data['SNOTEL'], merged_data['SNODAS'])
    mae = mean_absolute_error(merged_data['SNOTEL'], merged_data['SNODAS'])
    bias = np.mean(merged_data['SNODAS'] - merged_data['SNOTEL'])

    name = int(name) #This will help extract the columns from the dataframe
    elevation = sites_dict[name]['elev']

    elevation_values.append(elevation)
    corr_values.append(corr)
    rmse_values.append(rmse)
    mae_values.append(mae)
    bias_values.append(bias)

#Storing values in a dataframe
corr_df = pd.DataFrame({'Elevation': elevation_values, 'Correlation': corr_values})
rmse_df = pd.DataFrame({'Elevation': elevation_values, 'RMSE': rmse_values})
mae_df = pd.DataFrame({'Elevation': elevation_values, 'MAE': mae_values})
bias_df = pd.DataFrame({'Elevation': elevation_values, 'Bias': bias_values})

fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharey=True)

#Correlation Plot
axs[0, 0].scatter(corr_df['Correlation'], corr_df['Elevation'], alpha=0.7, s=10)
axs[0, 0].set_xlabel('Correlation', fontweight='bold')
axs[0, 0].set_ylabel('Elevation (m)', fontweight='bold')
axs[0, 0].set_title('Elevation control on Correlation', fontweight='bold')
axs[0, 0].grid(True)

#RMSE Plot
axs[0, 1].scatter(rmse_df['RMSE'], rmse_df['Elevation'], alpha=0.7, s=10)
axs[0, 1].set_xlabel('RMSE', fontweight='bold')
axs[0, 1].set_title('Elevation control on RMSE', fontweight='bold')
axs[0, 1].grid(True)

#MAE Plot
axs[1, 0].scatter(mae_df['MAE'], mae_df['Elevation'], alpha=0.7, s=10)
axs[1, 0].set_xlabel('MAE', fontweight='bold')
axs[1, 0].set_ylabel('Elevation (m)', fontweight='bold')
axs[1, 0].set_title('Elevation control on MAE', fontweight='bold')
axs[1, 0].grid(True)

#Bias plot
axs[1, 1].scatter(bias_df['Bias'], bias_df['Elevation'], alpha=0.7, s=10)
axs[1, 1].set_xlabel('Percentage Bias', fontweight='bold')
axs[1, 1].set_title('Elevation control on Percentage Bias', fontweight='bold')
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig('Comparison_metrics.png', dpi=300)
plt.show()
