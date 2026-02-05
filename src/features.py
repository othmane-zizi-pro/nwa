"""
Feature engineering functions for CLV prediction.
RFM (Recency, Frequency, Monetary) + Behavioral features.
"""
import pandas as pd
import numpy as np
from typing import Tuple
from datetime import datetime, timedelta


def calculate_rfm_features(
    df: pd.DataFrame,
    customer_id_col: str = "Customer ID",
    date_col: str = "InvoiceDate",
    amount_col: str = "TotalAmount",
    invoice_col: str = "Invoice",
    reference_date: datetime = None
) -> pd.DataFrame:
    """
    Calculate RFM features for each customer.

    Args:
        df: Transaction-level dataframe.
        customer_id_col: Column name for customer ID.
        date_col: Column name for transaction date.
        amount_col: Column name for transaction amount.
        invoice_col: Column name for invoice/transaction ID.
        reference_date: Reference date for recency calculation.

    Returns:
        DataFrame with RFM features per customer.
    """
    if reference_date is None:
        reference_date = df[date_col].max() + timedelta(days=1)

    # Aggregate at customer level
    rfm = df.groupby(customer_id_col).agg({
        date_col: lambda x: (reference_date - x.max()).days,  # Recency
        invoice_col: "nunique",  # Frequency
        amount_col: "sum"  # Monetary
    }).reset_index()

    rfm.columns = [customer_id_col, "Recency", "Frequency", "Monetary"]

    return rfm


def calculate_behavioral_features(
    df: pd.DataFrame,
    customer_id_col: str = "Customer ID",
    date_col: str = "InvoiceDate",
    amount_col: str = "TotalAmount",
    invoice_col: str = "Invoice",
    quantity_col: str = "Quantity",
    product_col: str = "StockCode",
    reference_date: datetime = None
) -> pd.DataFrame:
    """
    Calculate behavioral features for each customer.

    Args:
        df: Transaction-level dataframe.
        customer_id_col: Column name for customer ID.
        date_col: Column name for transaction date.
        amount_col: Column name for transaction amount.
        invoice_col: Column name for invoice/transaction ID.
        quantity_col: Column name for quantity.
        product_col: Column name for product code.
        reference_date: Reference date for tenure calculation.

    Returns:
        DataFrame with behavioral features per customer.
    """
    if reference_date is None:
        reference_date = df[date_col].max() + timedelta(days=1)

    # Tenure: days since first purchase
    tenure = df.groupby(customer_id_col)[date_col].min().reset_index()
    tenure["Tenure"] = (reference_date - tenure[date_col]).dt.days
    tenure = tenure[[customer_id_col, "Tenure"]]

    # Average time between purchases
    def avg_time_between_purchases(group):
        dates = group.sort_values()
        if len(dates) < 2:
            return 0
        diffs = dates.diff().dropna().dt.days
        return diffs.mean() if len(diffs) > 0 else 0

    avg_purchase_gap = df.groupby(customer_id_col)[date_col].apply(
        avg_time_between_purchases
    ).reset_index()
    avg_purchase_gap.columns = [customer_id_col, "AvgTimeBetweenPurchases"]

    # Number of unique products purchased
    unique_products = df.groupby(customer_id_col)[product_col].nunique().reset_index()
    unique_products.columns = [customer_id_col, "NumUniqueProducts"]

    # Average basket size (items per transaction)
    basket_size = df.groupby([customer_id_col, invoice_col])[quantity_col].sum().reset_index()
    avg_basket = basket_size.groupby(customer_id_col)[quantity_col].mean().reset_index()
    avg_basket.columns = [customer_id_col, "AvgBasketSize"]

    # Average order value
    order_value = df.groupby([customer_id_col, invoice_col])[amount_col].sum().reset_index()
    avg_order_value = order_value.groupby(customer_id_col)[amount_col].mean().reset_index()
    avg_order_value.columns = [customer_id_col, "AvgOrderValue"]

    # Number of unique countries (for B2B insights)
    # (keeping this simple - most customers have 1 country)

    # Merge all behavioral features
    behavioral = tenure.merge(avg_purchase_gap, on=customer_id_col, how="left")
    behavioral = behavioral.merge(unique_products, on=customer_id_col, how="left")
    behavioral = behavioral.merge(avg_basket, on=customer_id_col, how="left")
    behavioral = behavioral.merge(avg_order_value, on=customer_id_col, how="left")

    return behavioral


def create_customer_features(
    df: pd.DataFrame,
    customer_id_col: str = "Customer ID",
    date_col: str = "InvoiceDate",
    amount_col: str = "TotalAmount",
    invoice_col: str = "Invoice",
    quantity_col: str = "Quantity",
    product_col: str = "StockCode",
    reference_date: datetime = None
) -> pd.DataFrame:
    """
    Create all customer-level features (RFM + Behavioral).

    Args:
        df: Transaction-level dataframe.
        customer_id_col: Column name for customer ID.
        date_col: Column name for transaction date.
        amount_col: Column name for transaction amount.
        invoice_col: Column name for invoice/transaction ID.
        quantity_col: Column name for quantity.
        product_col: Column name for product code.
        reference_date: Reference date for calculations.

    Returns:
        DataFrame with all customer features.
    """
    rfm = calculate_rfm_features(
        df, customer_id_col, date_col, amount_col, invoice_col, reference_date
    )

    behavioral = calculate_behavioral_features(
        df, customer_id_col, date_col, amount_col, invoice_col,
        quantity_col, product_col, reference_date
    )

    # Merge RFM and behavioral features
    features = rfm.merge(behavioral, on=customer_id_col, how="left")

    return features


def split_data_temporal(
    df: pd.DataFrame,
    date_col: str = "InvoiceDate",
    observation_months: int = 12,
    prediction_months: int = 6
) -> Tuple[pd.DataFrame, pd.DataFrame, datetime, datetime]:
    """
    Split data into observation and prediction periods for CLV calculation.

    Args:
        df: Transaction-level dataframe.
        date_col: Column name for transaction date.
        observation_months: Number of months for feature calculation.
        prediction_months: Number of months for CLV target.

    Returns:
        Tuple of (observation_df, prediction_df, observation_end_date, prediction_end_date)
    """
    min_date = df[date_col].min()
    max_date = df[date_col].max()

    # Calculate split dates
    observation_end = min_date + pd.DateOffset(months=observation_months)
    prediction_end = observation_end + pd.DateOffset(months=prediction_months)

    # Ensure prediction_end doesn't exceed data
    if prediction_end > max_date:
        prediction_end = max_date
        observation_end = max_date - pd.DateOffset(months=prediction_months)

    # Split data
    observation_df = df[df[date_col] < observation_end]
    prediction_df = df[
        (df[date_col] >= observation_end) & (df[date_col] < prediction_end)
    ]

    return observation_df, prediction_df, observation_end, prediction_end


def calculate_clv_target(
    observation_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    customer_id_col: str = "Customer ID",
    amount_col: str = "TotalAmount"
) -> pd.DataFrame:
    """
    Calculate CLV target (total spend in prediction period) for customers in observation period.

    Args:
        observation_df: Transactions in observation period.
        prediction_df: Transactions in prediction period.
        customer_id_col: Column name for customer ID.
        amount_col: Column name for transaction amount.

    Returns:
        DataFrame with customer IDs and their CLV target.
    """
    # Get customers in observation period
    obs_customers = observation_df[customer_id_col].unique()

    # Calculate CLV (total spend in prediction period)
    clv = prediction_df.groupby(customer_id_col)[amount_col].sum().reset_index()
    clv.columns = [customer_id_col, "CLV"]

    # Add customers with 0 CLV (didn't purchase in prediction period)
    all_customers = pd.DataFrame({customer_id_col: obs_customers})
    clv = all_customers.merge(clv, on=customer_id_col, how="left")
    clv["CLV"] = clv["CLV"].fillna(0)

    return clv
