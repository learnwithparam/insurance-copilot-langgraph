"""SQLite customer + claims database for the insurance copilot.

The billing and claims agents call these helpers as tools. The schema is kept
intentionally tiny so learners can reason about it end to end.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

DB_PATH = os.getenv("INSURANCE_DB_PATH", "insurance.db")


SEED_CUSTOMERS: List[Dict[str, Any]] = [
    {
        "customer_id": "CUST-1001",
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "plan": "Auto Gold",
        "premium_monthly": 129.50,
        "balance_due": 0.00,
    },
    {
        "customer_id": "CUST-1002",
        "name": "Ben Rivera",
        "email": "ben@example.com",
        "plan": "Home Silver",
        "premium_monthly": 89.00,
        "balance_due": 89.00,
    },
    {
        "customer_id": "CUST-1003",
        "name": "Chen Wei",
        "email": "chen@example.com",
        "plan": "Auto Platinum",
        "premium_monthly": 184.75,
        "balance_due": 0.00,
    },
    {
        "customer_id": "CUST-1004",
        "name": "Divya Menon",
        "email": "divya@example.com",
        "plan": "Renters Basic",
        "premium_monthly": 22.10,
        "balance_due": 44.20,
    },
    {
        "customer_id": "CUST-1005",
        "name": "Ethan Park",
        "email": "ethan@example.com",
        "plan": "Health Family",
        "premium_monthly": 412.00,
        "balance_due": 0.00,
    },
]


SEED_CLAIMS: List[Dict[str, Any]] = [
    {"claim_id": "CLM-5001", "customer_id": "CUST-1001", "type": "collision", "status": "approved", "amount": 2400.00, "opened_at": "2026-01-12"},
    {"claim_id": "CLM-5002", "customer_id": "CUST-1001", "type": "glass", "status": "paid", "amount": 380.00, "opened_at": "2025-11-02"},
    {"claim_id": "CLM-5003", "customer_id": "CUST-1002", "type": "water damage", "status": "review", "amount": 5600.00, "opened_at": "2026-02-18"},
    {"claim_id": "CLM-5004", "customer_id": "CUST-1002", "type": "theft", "status": "denied", "amount": 1200.00, "opened_at": "2025-09-30"},
    {"claim_id": "CLM-5005", "customer_id": "CUST-1003", "type": "collision", "status": "approved", "amount": 7800.00, "opened_at": "2026-03-04"},
    {"claim_id": "CLM-5006", "customer_id": "CUST-1003", "type": "roadside", "status": "paid", "amount": 145.00, "opened_at": "2026-02-10"},
    {"claim_id": "CLM-5007", "customer_id": "CUST-1004", "type": "fire", "status": "review", "amount": 3200.00, "opened_at": "2026-03-22"},
    {"claim_id": "CLM-5008", "customer_id": "CUST-1005", "type": "hospital", "status": "approved", "amount": 14250.00, "opened_at": "2026-01-28"},
    {"claim_id": "CLM-5009", "customer_id": "CUST-1005", "type": "dental", "status": "paid", "amount": 620.00, "opened_at": "2025-12-15"},
    {"claim_id": "CLM-5010", "customer_id": "CUST-1001", "type": "towing", "status": "paid", "amount": 95.00, "opened_at": "2026-02-01"},
]


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def bootstrap_database() -> None:
    """Create tables and seed demo data on first run."""
    with _conn() as con:
        cur = con.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS customers (
                customer_id      TEXT PRIMARY KEY,
                name             TEXT NOT NULL,
                email            TEXT NOT NULL,
                plan             TEXT NOT NULL,
                premium_monthly  REAL NOT NULL,
                balance_due      REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS claims (
                claim_id    TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL REFERENCES customers(customer_id),
                type        TEXT NOT NULL,
                status      TEXT NOT NULL,
                amount      REAL NOT NULL,
                opened_at   TEXT NOT NULL
            );
            """
        )

        cur.execute("SELECT COUNT(*) AS n FROM customers")
        if cur.fetchone()["n"] == 0:
            cur.executemany(
                "INSERT INTO customers (customer_id, name, email, plan, premium_monthly, balance_due) "
                "VALUES (:customer_id, :name, :email, :plan, :premium_monthly, :balance_due)",
                SEED_CUSTOMERS,
            )
            cur.executemany(
                "INSERT INTO claims (claim_id, customer_id, type, status, amount, opened_at) "
                "VALUES (:claim_id, :customer_id, :type, :status, :amount, :opened_at)",
                SEED_CLAIMS,
            )


def get_customer(customer_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["claims"] = list_claims(customer_id)
        return record


def list_claims(customer_id: str) -> List[Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM claims WHERE customer_id = ? ORDER BY opened_at DESC",
            (customer_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def open_claim(customer_id: str, claim_type: str, amount: float) -> Dict[str, Any]:
    """Open a new claim in 'review' status and return it."""
    from datetime import date

    with _conn() as con:
        # pick next id
        row = con.execute("SELECT COUNT(*) AS n FROM claims").fetchone()
        claim_id = f"CLM-{5000 + row['n'] + 1}"
        record = {
            "claim_id": claim_id,
            "customer_id": customer_id,
            "type": claim_type,
            "status": "review",
            "amount": float(amount),
            "opened_at": date.today().isoformat(),
        }
        con.execute(
            "INSERT INTO claims (claim_id, customer_id, type, status, amount, opened_at) "
            "VALUES (:claim_id, :customer_id, :type, :status, :amount, :opened_at)",
            record,
        )
        return record


def reset_database() -> None:
    """Drop everything and re-seed. Used by POST /insurance-copilot/reset."""
    with _conn() as con:
        con.executescript("DROP TABLE IF EXISTS claims; DROP TABLE IF EXISTS customers;")
    bootstrap_database()
