"""Data loading and cleaning utilities for the Online Retail II dataset."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_FILENAME = "online_retail_II.xlsx"

ONLINE_RETAIL_XLSX_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
)
ONLINE_RETAIL_ZIP_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"


def ensure_raw_data(
    data_path: Optional[Path] = None,
    force_download: bool = False,
    timeout: int = 120,
) -> Path:
    """
    Ensure the raw Online Retail II file exists locally.

    Downloads from UCI if needed.
    """
    if data_path is None:
        data_path = RAW_DATA_DIR / RAW_DATA_FILENAME

    if data_path.exists() and not force_download:
        return data_path

    data_path.parent.mkdir(parents=True, exist_ok=True)

    last_error: Optional[Exception] = None
    for url in (ONLINE_RETAIL_XLSX_URL, ONLINE_RETAIL_ZIP_URL):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = response.read()

            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    xlsx_files = [
                        name for name in archive.namelist() if name.lower().endswith(".xlsx")
                    ]
                    if not xlsx_files:
                        raise RuntimeError(
                            "Zip downloaded successfully but no .xlsx file was found."
                        )
                    with archive.open(xlsx_files[0]) as src, data_path.open("wb") as dst:
                        dst.write(src.read())
            else:
                data_path.write_bytes(payload)

            return data_path
        except Exception as err:  # pragma: no cover - network-dependent
            last_error = err

    raise RuntimeError(
        "Failed to download Online Retail II data from all known sources."
    ) from last_error


def load_raw_data(
    data_path: Optional[Path] = None,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    """
    Load the Online Retail II dataset from Excel file.

    Args:
        data_path: Path to the Excel file. If None, uses default location.

    Returns:
        DataFrame with all sheets combined.
    """
    if data_path is None:
        data_path = RAW_DATA_DIR / RAW_DATA_FILENAME

    if not data_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"Raw data file not found at {data_path}. Set download_if_missing=True."
            )
        data_path = ensure_raw_data(data_path=data_path, force_download=False)
    excel_file = pd.ExcelFile(data_path)
    frames = [pd.read_excel(data_path, sheet_name=sheet) for sheet in excel_file.sheet_names]
    df = pd.concat(frames, ignore_index=True)

    # Normalize known alternate column names.
    df = df.rename(
        columns={
            "CustomerID": "Customer ID",
            "InvoiceNo": "Invoice",
            "UnitPrice": "Price",
        }
    )

    return df


def load_processed_data(filename: str = "cleaned_retail.csv") -> pd.DataFrame:
    """
    Load processed/cleaned data.

    Args:
        filename: Name of the processed file.

    Returns:
        DataFrame with cleaned data.
    """
    data_path = PROCESSED_DATA_DIR / filename
    parse_dates = ["InvoiceDate"] if filename in {"cleaned_retail.csv"} else None
    return pd.read_csv(data_path, parse_dates=parse_dates)


def load_customer_features(filename: str = "customer_features.csv") -> pd.DataFrame:
    """
    Load customer-level features (RFM + behavioral).

    Args:
        filename: Name of the features file.

    Returns:
        DataFrame with customer features.
    """
    data_path = PROCESSED_DATA_DIR / filename
    return pd.read_csv(data_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Online Retail II data.

    Steps:
    - Remove rows with missing CustomerID
    - Remove cancelled transactions (Invoice starts with 'C')
    - Remove rows with non-positive Quantity
    - Remove rows with non-positive Price
    - Convert data types

    Args:
        df: Raw dataframe.

    Returns:
        Cleaned dataframe.
    """
    df_clean = df.copy()

    required_columns = [
        "Invoice",
        "StockCode",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
    ]
    missing_cols = [col for col in required_columns if col not in df_clean.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in raw data: {missing_cols}")

    # Remove missing Customer ID
    df_clean = df_clean.dropna(subset=["Customer ID"])

    # Convert Customer ID to int
    df_clean["Customer ID"] = df_clean["Customer ID"].astype(int)

    # Remove cancelled transactions (Invoice starts with 'C')
    df_clean = df_clean[~df_clean["Invoice"].astype(str).str.startswith("C")]

    # Remove non-positive quantities
    df_clean = df_clean[df_clean["Quantity"] > 0]

    # Remove non-positive prices
    df_clean = df_clean[df_clean["Price"] > 0]

    # Convert Invoice to string
    df_clean["Invoice"] = df_clean["Invoice"].astype(str)

    # Ensure invoice date is datetime
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])

    # Calculate total amount per line item
    df_clean["TotalAmount"] = df_clean["Quantity"] * df_clean["Price"]

    return df_clean.reset_index(drop=True)
