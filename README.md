# HW_CodeMaintain
Hydrological Data Analysis Project
Overview
This project focuses on the analysis of hydrological data, specifically related to Snow Water Equivalent (SWE) from the SNODAS dataset. 
It includes the processing of geospatial data, extraction of time series information for specific locations, and a comparative analysis with ground-based SNOTEL measurements.
The goal is to explore the correlation, biases, and other metrics between SNODAS and SNOTEL data across different elevations.


Features
Importing and processing geospatial data using GeoPandas and Xarray.
Clipping SNODAS data to specific basins and time ranges.
Extracting time series information for SNOTEL sites and performing a comparative analysis.
Creating visualizations to illustrate the relationship between elevation and analysis metrics.


Installation
Clone the repository: git clone https://github.com/sghimi14/HW_CodeMaintain.git

Usage
Follow the steps outlined in the code to analyze hydrological data. Adjust file paths and configurations as needed.

Data Sources
SNODAS dataset: https://nsidc.org/data/g02158/versions/1
SNOTEL measurements: https://www.drought.gov/data-maps-tools/nrcs-snotel-and-snow-course-data
File Structure
Outputs_2018_2023/, Outputs_2010_2017/, Outputs_2004_2009/: Directories containing SNODAS NetCDF files.
Analysis_Basin_Shapefiles/: Shapefiles for CRB, Upper Basin, and Lower Basin.
snotel_metadata.csv: Metadata file for SNOTEL sites.
snotel_pivoted.csv: Pivoted SNOTEL data.
SNODAS_ts/: Directory to store time series CSV files for SNODAS data.


Contributing
Contributions are welcome! If you have ideas for improvements or new features, feel free to create issues or pull requests.
