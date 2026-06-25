# app/services/etl_service.py

"""
ETL Pipeline Service - Extract, Transform, Load.

Extract:
    Load raw CSV data using pandas.

Transform:
    - Normalize dates
    - Normalize currencies
    - Convert amounts to USD
    - Handle missing values
    - Validate required fields

Load:
    Upsert cleaned records into SQLite via OrderRepository.

Design decisions:
    - Batch processing for scalability
    - Row-level error tracking
    - Fixed currency conversion rates
    - Multiple date format support
    - Invalid amounts default to 0.0
    - Missing/unknown currency defaults to USD
"""

from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.core.config import settings
from app.core.exceptions import ETLProcessingError
from app.core.logging import get_logger
from app.repositories.order_repository import OrderRepository
from app.schemas.order import ETLRunResponse

logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

DATE_FORMATS: List[str] = [
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%m-%d-%Y",
    "%Y/%m/%d",
]

CURRENCY_RATES: Dict[str, float] = {
    "USD": 1.0,
    "EUR": settings.EUR_TO_USD_RATE,
}

INVALID_AMOUNT_VALUES = {
    "",
    "NA",
    "N/A",
    "/N/A",
    "NULL",
    "null",
    "None",
    "none",
    "nan",
    "NaN",
}


# ---------------------------------------------------------------------
# Date Normalization
# ---------------------------------------------------------------------


def normalize_date(raw_value: Any) -> Optional[date]:
    """
    Convert raw date values into a Python date.
    """
    if raw_value is None or pd.isna(raw_value):
        return None

    if isinstance(raw_value, datetime):
        return raw_value.date()

    if isinstance(raw_value, date):
        return raw_value

    raw_str = str(raw_value).strip()

    if not raw_str:
        return None

    for fmt in DATE_FORMATS:
        try:
            return pd.to_datetime(raw_str, format=fmt).date()
        except (ValueError, TypeError):
            continue

    try:
        return pd.to_datetime(raw_str).date()
    except Exception:
        return None


# ---------------------------------------------------------------------
# Amount Normalization
# ---------------------------------------------------------------------


def normalize_amount(raw_value: Any) -> Optional[float]:
    """
    Parse a numeric amount.
    """
    if raw_value is None or pd.isna(raw_value):
        return None

    raw_str = str(raw_value).strip()

    if raw_str in INVALID_AMOUNT_VALUES:
        return None

    try:
        value = float(raw_str.replace(",", "."))
        return value if value >= 0 else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------
# Currency Normalization
# ---------------------------------------------------------------------


def normalize_currency(raw_value: Any) -> str:
    """
    Normalize currency code.
    """
    if raw_value is None or pd.isna(raw_value):
        return "USD"

    raw_str = str(raw_value).strip().upper()

    if not raw_str:
        return "USD"

    return raw_str if raw_str in CURRENCY_RATES else "USD"


# ---------------------------------------------------------------------
# Currency Conversion
# ---------------------------------------------------------------------


def convert_to_usd(amount: float, currency: str) -> float:
    """
    Convert amount to USD.
    """
    rate = CURRENCY_RATES.get(currency, 1.0)
    return round(amount * rate, 2)


# ---------------------------------------------------------------------
# ETL Service
# ---------------------------------------------------------------------


class ETLService:
    """
    ETL orchestration service.
    """

    def __init__(
        self,
        repository: OrderRepository,
        batch_size: int = settings.ETL_BATCH_SIZE,
    ) -> None:
        self._repo = repository
        self._batch_size = batch_size

    async def run(self, file_path: str) -> ETLRunResponse:
        """
        Execute ETL pipeline for a CSV file.
        """
        start_time = time.perf_counter()

        path = Path(file_path)

        logger.info(
            "etl_run_started",
            file_path=str(path),
        )

        if not path.exists():
            raise ETLProcessingError(
                message=f"CSV file not found: {file_path}",
                file_path=file_path,
            )

        if path.suffix.lower() != ".csv":
            raise ETLProcessingError(
                message=f"Expected a .csv file, got: {path.suffix}",
                file_path=file_path,
            )

        try:
            df = pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                na_values=[""],
            )
        except Exception as exc:
            raise ETLProcessingError(
                message=f"Failed to read CSV file: {exc}",
                file_path=file_path,
            ) from exc

        rows_read = len(df)

        logger.info(
            "etl_csv_loaded",
            file_path=str(path),
            rows_read=rows_read,
        )

        cleaned_rows, row_errors = self._transform(
            df,
            file_path=str(path),
        )

        rows_loaded, rows_updated = await self._load_batches(
            cleaned_rows,
        )

        duration = time.perf_counter() - start_time

        rows_skipped = rows_read - len(cleaned_rows)

        logger.info(
            "etl_run_complete",
            file_path=str(path),
            rows_read=rows_read,
            rows_loaded=rows_loaded,
            rows_updated=rows_updated,
            rows_skipped=rows_skipped,
            duration_seconds=round(duration, 3),
            error_count=len(row_errors),
        )

        status = "success"

        if row_errors:
            status = "partial"

        if rows_loaded == 0 and rows_updated == 0 and rows_read > 0:
            status = "failed"

        return ETLRunResponse(
            status=status,
            file_path=str(path),
            rows_read=rows_read,
            rows_loaded=rows_loaded,
            rows_updated=rows_updated,
            rows_skipped=rows_skipped,
            duration_seconds=round(duration, 3),
            errors=row_errors[:100],
        )

    def _transform(
        self,
        df: pd.DataFrame,
        file_path: str = "",
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Transform raw dataframe into clean order records.
        """
        cleaned: List[Dict[str, Any]] = []
        errors: List[str] = []

        df.columns = [str(col).strip().lower() for col in df.columns]

        required_columns = {
            "order_id",
            "customer_id",
            "order_date",
            "amount",
            "currency",
        }

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ETLProcessingError(
                message=(
                    "CSV is missing required columns: "
                    f"{', '.join(sorted(missing_columns))}"
                ),
                file_path=file_path,
            )

        for idx, row in df.iterrows():
            row_num = idx + 2

            order_id = str(
                row.get("order_id", "")
            ).strip()

            if (
                not order_id
                or order_id.lower() in {"nan", "none"}
            ):
                errors.append(
                    f"Row {row_num}: Skipped - missing order_id"
                )
                continue

            customer_id = str(
                row.get("customer_id", "")
            ).strip()

            if (
                not customer_id
                or customer_id.lower() in {"nan", "none"}
            ):
                errors.append(
                    f"Row {row_num}: Skipped - missing customer_id "
                    f"(order_id={order_id})"
                )
                continue

            parsed_date = normalize_date(
                row.get("order_date")
            )

            if parsed_date is None:
                errors.append(
                    f"Row {row_num}: Skipped - unparseable "
                    f"order_date '{row.get('order_date')}' "
                    f"(order_id={order_id})"
                )
                continue

            amount = normalize_amount(
                row.get("amount")
            )

            if amount is None:
                errors.append(
                    f"Row {row_num}: amount "
                    f"'{row.get('amount')}' is invalid - "
                    f"defaulting to 0.0 "
                    f"(order_id={order_id})"
                )
                amount = 0.0

            currency = normalize_currency(
                row.get("currency")
            )

            amount_usd = convert_to_usd(
                amount,
                currency,
            )

            cleaned.append(
                {
                    "order_id": order_id,
                    "customer_id": customer_id,
                    "order_date": parsed_date,
                    "amount_usd": amount_usd,
                    "original_amount": amount,
                    "currency": currency,
                }
            )

        logger.info(
            "etl_transform_complete",
            total_rows=len(df),
            clean_rows=len(cleaned),
            skipped_rows=len(df) - len(cleaned),
            error_count=len(errors),
        )

        return cleaned, errors

    async def _load_batches(
        self,
        rows: List[Dict[str, Any]],
    ) -> Tuple[int, int]:
        """
        Load rows in batches.
        """
        total_inserted = 0
        total_updated = 0

        for start in range(
            0,
            len(rows),
            self._batch_size,
        ):
            batch = rows[start : start + self._batch_size]

            inserted, updated = await self._repo.upsert_batch(
                batch
            )

            total_inserted += inserted
            total_updated += updated

            logger.debug(
                "etl_batch_loaded",
                batch_start=start,
                batch_size=len(batch),
                total_processed=(
                    total_inserted + total_updated
                ),
            )

        return total_inserted, total_updated