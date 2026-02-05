"""
Data loading utilities for Online Retail II dataset.
"""
import pandas as pd
from pathlib import Path
from typing import Optional


def load_raw_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the Online Retail II dataset from Excel file.

    Args:
        data_path: Path to the Excel file. If None, uses default location.

    Returns:
        DataFrame with all sheets combined.
    """
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "raw" / "online_retail_II.xlsx"

    # Load both sheets (Year 2009-2010 and Year 2010-2011)
    df_2009_2010 = pd.read_excel(data_path, sheet_name="Year 2009-2010")
    df_2010_2011 = pd.read_excel(data_path, sheet_name="Year 2010-2011")

    # Combine sheets
    df = pd.concat([df_2009_2010, df_2010_2011], ignore_index=True)

    return df


def load_processed_data(filename: str = "cleaned_retail.csv") -> pd.DataFrame:
    """
    Load processed/cleaned data.

    Args:
        filename: Name of the processed file.

    Returns:
        DataFrame with cleaned data.
    """
    data_path = Path(__file__).parent.parent / "data" / "processed" / filename
    return pd.read_csv(data_path, parse_dates=["InvoiceDate"])


def load_customer_features(filename: str = "customer_features.csv") -> pd.DataFrame:
    """
    Load customer-level features (RFM + behavioral).

    Args:
        filename: Name of the features file.

    Returns:
        DataFrame with customer features.
    """
    data_path = Path(__file__).parent.parent / "data" / "processed" / filename
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

    # Remove missing CustomerID
    df_clean = df_clean.dropna(subset=["Customer ID"])

    # Convert CustomerID to int
    df_clean["Customer ID"] = df_clean["Customer ID"].astype(int)

    # Remove cancelled transactions (Invoice starts with 'C')
    df_clean = df_clean[~df_clean["Invoice"].astype(str).str.startswith("C")]

    # Remove non-positive quantities
    df_clean = df_clean[df_clean["Quantity"] > 0]

    # Remove non-positive prices
    df_clean = df_clean[df_clean["Price"] > 0]

    # Convert Invoice to string
    df_clean["Invoice"] = df_clean["Invoice"].astype(str)

    # Ensure InvoiceDate is datetime
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])

    # Calculate total amount per line item
    df_clean["TotalAmount"] = df_clean["Quantity"] * df_clean["Price"]

    return df_clean.reset_index(drop=True)
