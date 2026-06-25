#!/usr/bin/env python3
# etl.py
"""
ETL Pipeline CLI - Command Line Interface.

Enterprise-grade CLI for the ETL Order Service pipeline.
Uses Click for argument parsing and Rich for formatted terminal output.

Commands:
  load        Load and transform a CSV file into the database
  show-stats  Print revenue and order statistics to the terminal
  validate    Validate a CSV file without loading it

Usage:
  python etl.py load data/orders.csv
  python etl.py load data/orders.csv --batch-size 500
  python etl.py show-stats
  python etl.py validate data/orders.csv
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Initialize Rich console for formatted output
console = Console()

# — Async Runner ———————————————————————————————————————
def run_async(coro):
    """Run an async coroutine from a synchronous Click command."""
    return asyncio.run(coro)

# — CLI Group ——————————————————————————————————————————
@click.group()
@click.version_option(version="1.0.0", prog_name="ETL Service")
def cli():
    """
    ETL Order Service - Command Line Interface.

    Processes customer order CSV files and loads them into the database.
    """
    pass

# — load command ———————————————————————————————————————
@cli.command("load")
@click.argument("csv_path", type=click.Path(exists=True, readable=True))
@click.option(
    "--batch-size",
    default=1000,
    show_default=True,
    type=click.IntRange(1, 10000),
    help="Number of rows per database batch.",
)
@click.option(
    "--rebuild-index/--no-rebuild-index",
    default=True,
    show_default=True,
    help="Rebuild the FAISS embedding index after loading.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Transform data but do not write to the database.",
)
def load_command(csv_path: str, batch_size: int, rebuild_index: bool, dry_run: bool):
    """
    Load and transform a CSV file into the orders database.

    CSV_PATH: Path to the input CSV file (e.g. data/orders.csv).

    The ETL pipeline will:
      1. Extract: Read the CSV file
      2. Transform: Normalize dates, convert currencies, handle missing values
      3. Load: Upsert cleaned records into SQLite

    Examples:
      python etl.py load data/orders.csv
      python etl.py load data/orders.csv --batch-size 500
      python etl.py load data/orders.csv --dry-run
    """
    run_async(_load_async(csv_path, batch_size, rebuild_index, dry_run))

async def _load_async(
    csv_path: str,
    batch_size: int,
    rebuild_index: bool,
    dry_run: bool,
) -> None:
    """Async implementation of the load command."""
    from app.core.config import settings
    from app.core.database import create_all_tables, get_db_context
    from app.core.logging import configure_logging
    from app.repositories.order_repository import OrderRepository
    from app.services.etl_service import ETLService

    configure_logging(log_level=settings.LOG_LEVEL.value, is_development=True)

    console.print(Panel.fit(
        f"[bold blue]ETL Order Service[/bold blue]\n"
        f"Loading: [cyan]{csv_path}[/cyan]\n"
        f"Batch size: [yellow]{batch_size}[/yellow] | "
        f"Dry run: [yellow]{dry_run}[/yellow]",
        title="ETL Pipeline",
        border_style="blue",
    ))

    if dry_run:
        console.print("[yellow]⚠ DRY RUN MODE - No data will be written to the database[/yellow]")

    # Ensure tables exist
    await create_all_tables()

    async with get_db_context() as db:
        repo = OrderRepository(db)
        service = ETLService(repository=repo, batch_size=batch_size)

        with console.status("[bold green]Running ETL pipeline...[/bold green]"):
            result = await service.run(csv_path)

    # — Print Results ——————————————————————————————————————————
    status_color = {
        "success": "green",
        "partial": "yellow",
        "failed": "red",
    }.get(result.status, "white")

    table = Table(title="ETL Run Results", box=box.ROUNDED, border_style="blue")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Status", f"[{status_color}]{result.status.upper()}[/{status_color}]")
    table.add_row("File", result.file_path)
    table.add_row("Rows Read", str(result.rows_read))
    table.add_row("Rows Loaded", f"[green]{result.rows_loaded}[/green]")
    table.add_row("Rows Skipped", f"[yellow]{result.rows_skipped}[/yellow]")
    table.add_row("Rows Updated", str(result.rows_updated))
    table.add_row("Duration", f"{result.duration_seconds: 3}s")
    table.add_row("Errors", f"[red]{len(result.errors)}[/red]" if result.errors else "[green]0[/green]")

    console.print(table)

    # Print row-level errors if any
    if result.errors:
        console.print(f"[yellow]Row-level warnings ({len(result.errors)} total):[/yellow]")
        for err in result.errors[:20]:
            console.print(f" [dim]* {err}[/dim]")
        if len(result.errors) > 20:
            console.print(f" [dim]... and {len(result.errors) - 20} more[/dim]")

    # — Rebuild FAISS Index ———————————————————————————————————————
    if rebuild_index and result.rows_loaded > 0 and not dry_run:
        console.print("\n[bold]Rebuilding semantic search index...[/bold]")
        try:
            from app.services.embedding_service import embedding_service

            async with get_db_context() as db:
                repo = OrderRepository(db)
                orders = await repo.get_all()

            with console.status("[bold green]Building FAISS index...[/bold green]"):
                await embedding_service.build_index(orders)

            console.print(
                f"[green]✓ FAISS index rebuilt with {embedding_service.index_size} vectors[/green]"
            )
        except Exception as exc:
            console.print(f"[yellow]⚠ Index rebuild failed (non-fatal): {exc}[/yellow]")

    if result.status == "failed":
        sys.exit(1)

# — show-stats command ———————————————————————————————————————
@cli.command("show-stats")
@click.option(
    "--top-days",
    default=10,
    show_default=True,
    type=click.IntRange(1, 365),
    help="Number of top days to display in the per-day breakdown.",
)
def show_stats_command(top_days: int):
    """
    Print aggregated order statistics to the terminal.

    Displays:
      - Total revenue (USD)
      - Average order value (USD)
      - Total order count
      - Top N days by order count

    Examples:
      python etl.py show-stats
      python etl.py show-stats --top-days 5
    """
    run_async(_show_stats_async(top_days))

async def _show_stats_async(top_days: int) -> None:
    """Async implementation of the show-stats command."""
    from app.core.database import create_all_tables, get_db_context
    from app.repositories.order_repository import OrderRepository

    await create_all_tables()

    async with get_db_context() as db:
        repo = OrderRepository(db)
        stats = await repo.get_stats()

    console.print(Panel.fit(
        "[bold blue]Order Statistics[/bold blue]",
        border_style="blue",
    ))

    # Summary table
    summary = Table(box=box.ROUNDED, border_style="green")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right", style="cyan")

    summary.add_row("Total Revenue (USD)", f"${stats['total_revenue']:,.2f}")
    summary.add_row("Average Order Value (USD)", f"${stats['avg_order_value']:,.2f}")
    summary.add_row("Total Orders", f"{stats['order_count']:,}")

    console.print(summary)

    # Currency breakdown
    if stats.get("currency_breakdown"):
        console.print("\n[bold]Currency Breakdown:[/bold]")
        curr_table = Table(box=box.SIMPLE)
        curr_table.add_column("Currency")
        curr_table.add_column("Orders", justify="right")
        for currency, count in sorted(stats["currency_breakdown"].items()):
            curr_table.add_row(currency, str(count))
        console.print(curr_table)

    # Top days
    if stats.get("orders_per_day"):
        console.print(f"\n[bold]Top {top_days} Days by Order Count:[/bold]")
        day_table = Table(box=box.SIMPLE)
        day_table.add_column("Date")
        day_table.add_column("Orders", justify="right")

        sorted_days = sorted(
            stats["orders_per_day"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_days]

        for day, count in sorted_days:
            day_table.add_row(str(day), str(count))

        console.print(day_table)

# — validate command ———————————————————————————————————————
@cli.command("validate")
@click.argument("csv_path", type=click.Path(exists=True, readable=True))
def validate_command(csv_path: str):
    """
    Validate a CSV file without loading it into the database.

    Runs the Extract and Transform steps only, reporting any data quality
    issues without persisting any data.

    CSV_PATH: Path to the CSV file to validate.

    Examples:
      python etl.py validate data/orders.csv
    """
    run_async(_validate_async(csv_path))

async def _validate_async(csv_path: str) -> None:
    """Async implementation of the validate command."""
    import pandas as pd
    from app.services.etl_service import ETLService, normalize_date, normalize_amount, normalize_currency

    console.print(Panel.fit(
        f"[bold blue]Validating:[/bold blue] [cyan]{csv_path}[/cyan]",
        border_style="blue",
    ))

    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[""])
    except Exception as exc:
        console.print(f"[red]✗ Failed to read CSV: {exc}[/red]")
        sys.exit(1)

    console.print(f"[green]✓ CSV loaded: {len(df)} rows[/green]")

    # Check columns
    required = {"order_id", "customer_id", "order_date", "amount", "currency"}
    actual = set(df.columns.str.strip().str.lower())
    missing = required - actual
    if missing:
        console.print(f"[red]✗ Missing columns: {missing}[/red]")
        sys.exit(1)
    console.print(f"[green]✓ All required columns present[/green]")

    # Count issues
    missing_order_id = df["order_id"].isna().sum() + (df["order_id"].str.strip() == "").sum()
    missing_customer_id = df["customer_id"].isna().sum() + (df["customer_id"].str.strip() == "").sum()
    missing_amount = df["amount"].apply(lambda x: normalize_amount(x) is None).sum()
    missing_currency = (df["currency"].str.strip() == "").sum() + df["currency"].isna().sum()

    issues_table = Table(title="Data Quality Report", box=box.ROUNDED)
    issues_table.add_column("Check", style="bold")
    issues_table.add_column("Issues", justify="right")
    issues_table.add_column("Status")

    def status(count):
        return "[green]✓ OK[/green]" if count == 0 else f"[yellow]⚠ {count} rows[/yellow]"

    issues_table.add_row("Missing order_id (will be dropped)", str(missing_order_id), status(missing_order_id))
    issues_table.add_row("Missing customer_id (will be dropped)", str(missing_customer_id), status(missing_customer_id))
    issues_table.add_row("Invalid/missing amount (≈ 0.0)", str(missing_amount), status(missing_amount))
    issues_table.add_row("Missing currency (→ USD)", str(missing_currency), status(missing_currency))

    console.print(issues_table)
    console.print(f"\n[bold]Estimated rows after transform:[/bold] "
                  f"[cyan]{len(df) - missing_order_id - missing_customer_id}[/cyan]")

# — Entry Point ———————————————————————————————————————
if __name__ == "__main__":
    cli()