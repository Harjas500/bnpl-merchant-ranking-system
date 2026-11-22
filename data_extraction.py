import requests
import xml.etree.ElementTree as ET
import os
import pandas as pd
import io
import re
from pathlib import Path
from zipfile import ZipFile
from pathlib import Path


def download_abs_data(
    url: str,
    savefile: str,
    metadata_file: str = None,
    headers: dict = None,
    start_period: str = "2020"
) -> pd.DataFrame:
    """
    Downloads data (with all dimensions per observation) from the ABS API.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(savefile), exist_ok=True)
    
    if headers is None:
        headers = {
            'Accept': 'text/csv',
            'User-Agent': 'Mozilla/5.0'
        }
    params = {
        "startPeriod": start_period,
        "dimensionAtObservation": "AllDimensions"
    }
    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()

    with open(savefile, "wb") as f:
        f.write(resp.content)
        
    # Read the CSV into DataFrame
    df = pd.read_csv(savefile)

    # Get unique measures from the data
    measures = df['MEASURE'].unique().tolist()
    
    # Get measure titles from metadata
    measure_dict = get_measure_titles(metadata_file, measures)
    
    # Add descriptions column
    df['MEASURE_DESC'] = df['MEASURE'].map(measure_dict)
    
    # Save enhanced version
    df.to_csv(savefile, index=False)
    print(f"Data saved to {savefile}")

    return df

def download_metadata(
    url: str,
    savefile: str = None,
    headers: dict = None
) -> bytes:
    """
    Downloads content from the given URL and optionally saves it to a file.
    """
    # Default header to get XML metadata, unless overridden
    if headers is None:
        headers = {
            'Accept': 'application/xml',
            'User-Agent': 'Mozilla/5.0'
        }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    content = response.content
    
    if savefile:
        savepath = Path(savefile)
        savepath.parent.mkdir(parents=True, exist_ok=True)  # <--- ensure folder exists
        with open(savepath, "wb") as f:
            f.write(content)
        print(f"Content saved to {savefile}")
    
    return content

def get_measure_titles(xml_file: str, measure_codes: list) -> dict:
    """
    Get the descriptive titles for a list of measure codes from ABS metadata XML.
    Helper function for the main download function.
    """
    # Define the XML namespaces
    namespaces = {
        'structure': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure',
        'common': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common'
    }
    
    # Parse XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Dictionary to store results
    measure_dict = {}
    
    # Find all Code elements and check if their id matches any of our measure codes
    for code in root.findall('.//structure:Code', namespaces):
        code_id = code.get('id')
        if code_id in measure_codes:
            name = code.find('.//common:Name', namespaces)
            if name is not None:
                measure_dict[code_id] = name.text
    
    return measure_dict

def download_and_extract_abs_zip(
    url: str,
    raw_dir: str = "data/raw",
    overwrite: bool = False,
    timeout: int = 60,
) -> Path:
    """
    Download a ZIP from an ABS URL and extract it into a dedicated folder under data/raw.

    - Creates data/raw if it doesn't exist.
    - The extract folder name is derived from the ZIP filename (without .zip).
    - If overwrite=False and the target folder exists, it will skip extraction.
    - Implements safe extraction (prevents zip-slip).

    Returns:
        Path: The path to the extracted folder.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    # Derive a sensible local filename from the URL
    fname = url.split("/")[-1] or "abs_download.zip"
    # Keep it clean: replace any weird chars with underscores
    fname = re.sub(r"[^A-Za-z0-9._-]", "_", fname)
    zip_path = raw_path / fname

    # Download (streaming) to memory, then write to disk (or straight to disk if preferred)
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()

    # Write to disk
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 128):
            if chunk:
                f.write(chunk)

    # Decide extract folder (use the filename stem)
    extract_dir = raw_path / Path(fname).stem
    if extract_dir.exists() and not overwrite:
        # Already extracted; return existing folder
        return extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Safe extraction
    def _is_within_directory(directory: Path, target: Path) -> bool:
        # Avoid zip-slip by ensuring target path stays within directory
        try:
            directory = directory.resolve()
            target = target.resolve()
            return str(target).startswith(str(directory) + os.sep) or str(target) == str(directory)
        except FileNotFoundError:
            # If not yet on disk, approximate using pure path logic
            return str(target).startswith(str(directory))

    with ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            # Normalize member path
            member_path = Path(member.filename)
            # Skip directory traversal attempts
            dest_path = extract_dir / member_path
            if not _is_within_directory(extract_dir, dest_path):
                raise RuntimeError(f"Blocked suspicious path in ZIP: {member.filename}")
            if member.is_dir():
                dest_path.mkdir(parents=True, exist_ok=True)
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())

    return extract_dir

def curated_df(census_columns: dict):
    os.makedirs("data/curated", exist_ok=True)
    result_df = pd.DataFrame({'postcode': [f"{i:04d}" for i in range(200, 9945)]})

    for packet, columns in census_columns.items():
        file_path = f"data/raw/2021_GCP_POA_for_AUS_short-header/2021 Census GCP Postal Areas for AUS/2021Census_{packet}_AUST_POA.csv"
        if os.path.exists(file_path):
            # Loading and processing each CSV file
            df = pd.read_csv(file_path, usecols=columns + ['POA_CODE_2021'])
            df['postcode'] = df['POA_CODE_2021'].apply(lambda x: f"{int(str(x).replace('POA', '')):04d}")
            df = df.drop(columns=['POA_CODE_2021'])
            
        # Merge each column into result_df
        result_df = result_df.merge(df, on='postcode', how='left')

    percent_cols = ["High_yr_schl_comp_Yr_12_eq_P", "Tert_Tot_Tert_P"]
    denom = pd.to_numeric(result_df['Tot_P_P'], errors='coerce')
    for col in percent_cols:
        if col in result_df.columns:
            numer = pd.to_numeric(result_df[col], errors='coerce')
            result_df[col] = (numer / denom) * 100
        else:
            print(f"Column {col} not found; skipping.")


    return result_df


