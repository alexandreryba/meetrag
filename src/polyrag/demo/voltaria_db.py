"""Deterministic generator for the analytics database of Voltaria Energy (a fictional
European electricity retailer). Offer prices match the product documentation in
``data/corpus/products`` so that cross-source questions have consistent answers."""

from __future__ import annotations

import math
import random
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

SNAPSHOT_DATE = date(2026, 9, 30)
FIRST_BILLED_MONTH = date(2025, 1, 1)

SCHEMA = """
CREATE TABLE offers (
    offer_id TEXT PRIMARY KEY,           -- short code, e.g. 'FIXE3'
    name TEXT NOT NULL,                  -- commercial name
    pricing TEXT NOT NULL,               -- 'fixed' | 'indexed' | 'time-of-use'
    segment TEXT NOT NULL,               -- target segment: 'residential' | 'professional'
    price_eur_kwh REAL NOT NULL,         -- price per kWh incl. taxes (peak price for time-of-use)
    offpeak_price_eur_kwh REAL,          -- off-peak price per kWh, time-of-use offers only
    monthly_fee_eur REAL NOT NULL,       -- monthly subscription incl. taxes
    launched_on DATE NOT NULL
);
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    segment TEXT NOT NULL,               -- 'residential' | 'professional'
    region TEXT NOT NULL,                -- French administrative region
    offer_id TEXT NOT NULL REFERENCES offers (offer_id),
    signup_date DATE NOT NULL,
    status TEXT NOT NULL,                -- 'active' | 'churned'
    churn_date DATE                      -- NULL while the customer is active
);
CREATE TABLE monthly_consumption (
    customer_id INTEGER NOT NULL REFERENCES customers (customer_id),
    month TEXT NOT NULL,                 -- billing month 'YYYY-MM' (from 2025-01 to 2026-09)
    kwh REAL NOT NULL,                   -- electricity consumed in the month
    amount_billed_eur REAL NOT NULL,     -- amount invoiced for the month incl. taxes
    PRIMARY KEY (customer_id, month)
);
CREATE TABLE support_tickets (
    ticket_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers (customer_id),
    opened_at DATE NOT NULL,
    category TEXT NOT NULL,              -- 'billing' | 'contract' | 'meter' | 'outage' | 'move'
    priority TEXT NOT NULL,              -- 'P1' (critical) | 'P2' | 'P3'
    status TEXT NOT NULL,                -- 'open' | 'closed'
    closed_at DATE,                      -- NULL while open
    csat INTEGER                         -- satisfaction 1-5 after closure, NULL if no answer
);
"""


@dataclass(frozen=True)
class Offer:
    offer_id: str
    name: str
    pricing: str
    segment: str
    price: float
    offpeak_price: float | None
    monthly_fee: float
    launched_on: date
    monthly_churn: float


OFFERS = (
    Offer("ESSENTIEL", "Voltaria Essentielle", "indexed", "residential",
          0.1940, None, 11.80, date(2021, 6, 1), 0.0120),
    Offer("VERTE", "Voltaria Verte", "indexed", "residential",
          0.2050, None, 12.90, date(2022, 3, 1), 0.0060),
    Offer("FIXE3", "Voltaria Fixe 3 ans", "fixed", "residential",
          0.2016, None, 12.40, date(2023, 1, 15), 0.0040),
    Offer("DRIVE", "Voltaria Drive", "time-of-use", "residential",
          0.2290, 0.1350, 14.50, date(2024, 4, 2), 0.0050),
    Offer("PROFLEX", "Voltaria Pro Flex", "indexed", "professional",
          0.1890, None, 29.00, date(2022, 9, 12), 0.0080),
)  # fmt: skip

REGIONS = {
    "Auvergne-Rhône-Alpes": 0.22,
    "Île-de-France": 0.20,
    "Occitanie": 0.12,
    "Nouvelle-Aquitaine": 0.11,
    "Provence-Alpes-Côte d'Azur": 0.10,
    "Grand Est": 0.09,
    "Hauts-de-France": 0.09,
    "Bretagne": 0.07,
}

RESIDENTIAL_MIX = {"ESSENTIEL": 0.30, "VERTE": 0.28, "FIXE3": 0.27, "DRIVE": 0.15}
TICKET_CATEGORIES = {"billing": 0.38, "contract": 0.18, "meter": 0.16, "outage": 0.14, "move": 0.14}
MEAN_RESOLUTION_DAYS = {"billing": 6.0, "contract": 3.0, "meter": 4.0, "outage": 1.5, "move": 5.0}


def _choice(rng: random.Random, weights: dict[str, float]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def _add_months(day: date, months: int) -> date:
    year, month = divmod(day.month - 1 + months, 12)
    return date(day.year + year, month + 1, 1)


def _random_day(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, max((end - start).days, 0)))


def build_database(path: Path, *, n_customers: int = 2400, seed: int = 42) -> dict[str, int]:
    """(Re)create the SQLite database at ``path``. Returns the row count per table."""
    rng = random.Random(seed)  # noqa: S311 (synthetic data, not security-sensitive)
    offers = {offer.offer_id: offer for offer in OFFERS}
    customers: list[tuple[object, ...]] = []
    consumption: list[tuple[object, ...]] = []
    tickets: list[tuple[object, ...]] = []

    for customer_id in range(1, n_customers + 1):
        professional = rng.random() < 0.12
        segment = "professional" if professional else "residential"
        offer_id = (
            ("PROFLEX" if rng.random() < 0.9 else "VERTE")
            if professional
            else _choice(rng, RESIDENTIAL_MIX)
        )
        offer = offers[offer_id]
        signup = _random_day(rng, offer.launched_on, date(2026, 8, 31))

        churn_date: date | None = None
        month = _add_months(signup, 1)
        while month <= SNAPSHOT_DATE:
            if rng.random() < offer.monthly_churn:
                churn_date = _random_day(rng, month, min(_add_months(month, 1), SNAPSHOT_DATE))
                break
            month = _add_months(month, 1)

        customers.append((
            customer_id, segment, _choice(rng, REGIONS), offer_id, signup.isoformat(),
            "churned" if churn_date else "active",
            churn_date.isoformat() if churn_date else None,
        ))  # fmt: skip

        base_kwh = (1450.0 if professional else 380.0) * rng.uniform(0.6, 1.5)
        if offer_id == "DRIVE":
            base_kwh += 120.0  # electric-vehicle charging
        month = FIRST_BILLED_MONTH
        while month <= SNAPSHOT_DATE:
            month_end = _add_months(month, 1) - timedelta(days=1)
            if signup <= month_end and (churn_date is None or churn_date >= month):
                season = 1 + 0.35 * math.cos(2 * math.pi * (month.month - 1) / 12)
                kwh = round(base_kwh * season * rng.uniform(0.9, 1.1), 1)
                if offer.offpeak_price is not None:
                    energy_cost = kwh * (0.55 * offer.offpeak_price + 0.45 * offer.price)
                else:
                    energy_cost = kwh * offer.price
                amount = round(energy_cost + offer.monthly_fee, 2)
                consumption.append((customer_id, month.strftime("%Y-%m"), kwh, amount))

                if rng.random() < 0.06:
                    tickets.append(_ticket(rng, len(tickets) + 1, customer_id, month, month_end))
            month = _add_months(month, 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(SCHEMA)
        connection.executemany(
            "INSERT INTO offers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (o.offer_id, o.name, o.pricing, o.segment, o.price, o.offpeak_price,
                 o.monthly_fee, o.launched_on.isoformat())
                for o in OFFERS
            ],
        )  # fmt: skip
        connection.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?)", customers)
        connection.executemany("INSERT INTO monthly_consumption VALUES (?, ?, ?, ?)", consumption)
        connection.executemany(
            "INSERT INTO support_tickets VALUES (?, ?, ?, ?, ?, ?, ?, ?)", tickets
        )
        connection.commit()
    return {
        "offers": len(OFFERS),
        "customers": len(customers),
        "monthly_consumption": len(consumption),
        "support_tickets": len(tickets),
    }


def _ticket(
    rng: random.Random, ticket_id: int, customer_id: int, month: date, month_end: date
) -> tuple[object, ...]:
    category = _choice(rng, TICKET_CATEGORIES)
    opened = _random_day(rng, month, min(month_end, SNAPSHOT_DATE))
    if category == "outage":
        priority = _choice(rng, {"P1": 0.2, "P2": 0.6, "P3": 0.2})
    else:
        priority = _choice(rng, {"P1": 0.02, "P2": 0.30, "P3": 0.68})
    resolution_days = math.ceil(rng.expovariate(1 / MEAN_RESOLUTION_DAYS[category]))
    closed = opened + timedelta(days=resolution_days)
    if closed > SNAPSHOT_DATE:
        return (ticket_id, customer_id, opened.isoformat(), category, priority, "open", None, None)
    csat: int | None = None
    if rng.random() < 0.7:
        expected = 4.6 - 0.12 * resolution_days
        csat = max(1, min(5, round(rng.gauss(expected, 0.8))))
    return (
        ticket_id, customer_id, opened.isoformat(), category, priority, "closed",
        closed.isoformat(), csat,
    )  # fmt: skip
