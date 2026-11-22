from data_extraction import (
    curated_df,
    download_and_extract_abs_zip
)
import pandas as pd
import os

# Data Paths (just have to input DATA_BASE. Make sure you provide a 'data' folder)
DATA_BASE = "/Users/yuanern/Documents/unimelb/Applied Data Science/Assignment 2/data"
DATA_GIVEN = f"{DATA_BASE}/given"
DATA_RAW = f"{DATA_BASE}/raw"
DATA_CURATED = f"{DATA_BASE}/curated"

# Making sure the directories exist
os.makedirs(DATA_BASE, exist_ok=True)
os.makedirs(DATA_GIVEN, exist_ok=True)
os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_CURATED, exist_ok=True)

print(f"Directories created: {DATA_BASE}, \n{DATA_GIVEN}, \n{DATA_RAW}, \n{DATA_CURATED}")

url = "https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_POA_for_AUS_short-header.zip"
out_dir = download_and_extract_abs_zip(url, raw_dir="data/raw", overwrite=False)

# Loading specific columns from the extracted data
census_columns = {
    "G01": ["High_yr_schl_comp_Yr_12_eq_P", "Tot_P_P"],
    "G02": ["Median_mortgage_repay_monthly", "Median_tot_prsnl_inc_weekly", "Median_rent_weekly",
            "Median_tot_fam_inc_weekly", "Average_num_psns_per_bedroom", "Median_tot_hhd_inc_weekly"],
    "G15": ["Tert_Tot_Tert_P"],
    "G21B": ["TPI_wk_Tot_Total"],
    "G46B": ["P_Tot_LF_Tot", "P_Not_in_LF_Tot", "P_LFS_NS_Tot"]            
}

external_data = curated_df(census_columns)

# Joining with existing consumer data via postcode
consumer_df = pd.read_csv("data/curated/consumer_curated.csv")
consumer_df['postcode'] = consumer_df['postcode'].astype(str)
external_data['postcode'] = external_data['postcode'].astype(str)
merged_df = pd.merge(consumer_df, external_data, on='postcode', how='inner')
merged_df.to_csv("data/curated/consumer_curated_with_postcode.csv", index=False)

print("Merged data saved to data/curated/consumer_curated_with_postcode.csv")

