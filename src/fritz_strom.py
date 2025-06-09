#!/usr/bin/env python3
# vim: set et :
"""
Parse CSV energy consumption report from FRITZ!Smart Energy 200.
"""

from __future__ import annotations

import csv
import re
import sys
from argparse import ArgumentParser, FileType, Namespace
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar, Self

from intervaltree import IntervalTree
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

RE_FN1 = re.compile(
    r"""
    Hahn_Strom_
    (?P<d>\d{2})
    [.]
    (?P<m>\d{2})
    [.]
    (?P<Y>\d{4})
    _
    (?P<H>\d{2})
    -
    (?P<M>\d{2})
    _
    (?P<span>24h|week|month|2years)
    [.]csv""",
    re.VERBOSE,
)
RE_FN2 = re.compile(
    r"""
    (?P<Y>\d{4})
    (?P<m>\d{2})
    (?P<d>\d{2})
    _
    (?P<H>\d{2})
    (?P<M>\d{2})
    (?P<S>\d{2})
    _
    id\d+
    _
    1?  # optional prefix of "month"
    (?P<span>24h|week|month|2years)
    [.]csv""",
    re.VERBOSE,
)


@dataclass(slots=True)
class Values:
    usage: float
    price: float
    co2: float

    def __isub__(self, other: Self) -> Self:
        self.usage -= other.usage
        self.price -= other.price
        self.co2 -= other.co2
        return self

    def __str__(self) -> str:
        return (
            f"{self.usage:10.0f}"  # W
            f"  {self.price:7.2f}"  # €
            f"  {self.co2:7.3f}"  # kg CO₂
        )


@dataclass(frozen=True, slots=True)
class Row:
    date: str
    values: Values


@dataclass(slots=True)
class Node:
    start: datetime  # ascending
    end: datetime  # descending
    values: Values
    ts: datetime = field(repr=False)
    span: timedelta = field(init=False)

    def __post_init__(self) -> None:
        self.span = self.delta

    def __str__(self) -> str:
        span = self.end - self.start
        h, m = divmod(span.seconds // 60, 60)
        return (
            f"{self.start:%Y-%m-%d %H:%M}"
            f"  {span.days:2}d{h:2}h{m:2}m"
            f"  {self.values.usage:10.0f}"  # W
            # f"  {self.values.price:7.2f}"  # €
            # f"  {self.values.co2:7.3f}"  # kg CO₂
        )

    def __isub__(self, other: Self) -> Self:
        self.values -= other.values
        self.span -= other.span
        return self

    @property
    def delta(self) -> timedelta:
        return self.end - self.start


class Parser:
    KEY: str  # filename
    SPAN: str  # header

    @classmethod
    def detect(cls, fn: Path) -> Parser:
        if not (m := (RE_FN1.fullmatch(fn.name) or RE_FN2.fullmatch(fn.name))):
            raise ValueError(fn.name)

        last = datetime(
            int(m["Y"]),
            int(m["m"]),
            int(m["d"]),
            int(m["H"]),
            int(m["M"]),
            int(m.groupdict().get("S", 0)),
        )
        parser: dict[str, type[Parser]] = {
            parser.KEY: parser for parser in cls.__subclasses__()
        }
        span = parser[m["span"]]
        print(f"FN: {last} {span}", file=sys.stderr)
        return span(last)

    def __init__(self, last: datetime) -> None:
        self.last = self.ts = last.replace(second=0, microsecond=0)

    def parse_header(self, row: list[str]) -> None:
        (
            date_str,
            usage_str,
            usage_unit,
            price_str,
            price_unit,
            co2_str,
            co2_unit,
            empty,
            view,
            *rest,
        ) = row
        assert date_str in {"Datum/Uhrzeit", "Datum/Zeit"}, row
        assert usage_str in {"Verbrauchswert", "Energie"}, row
        assert usage_unit == "Einheit", row
        assert price_str in {"Verbrauch in Euro", "Energie in Euro"}, row
        assert price_unit == "Einheit", row
        assert co2_str == "CO2-Ausstoss", row
        assert co2_unit == "Einheit", row
        assert empty == "", row
        assert view == "Ansicht:", row

        if rest[0] == "Datum":
            date_str, empty, span, ts_str = rest
        else:
            span, empty, date_str, ts_str = rest

        assert date_str == "Datum", rest
        assert empty == "", rest
        assert span == self.SPAN, row

        if "-" in ts_str:
            self.ts = datetime.strptime(ts_str, "%d.%m.%Y %H-%M Uhr")
        elif ":" in ts_str:
            self.ts = datetime.strptime(ts_str, "%d.%m.%Y %H:%M Uhr")
        else:
            raise ValueError(ts_str)
        assert self.ts == self.last, f"{self.ts=} {self.last=}"

    def parse_row(self, row: list[str]) -> Row:
        date_str, usage_str, usage_unit, price_str, price_unit, co2_str, co2_unit = row
        usage = float(usage_str.replace(",", ".")) * (
            1_000 if usage_unit == "kWh" else 1
        )
        assert usage_unit in {"Wh", "kWh"}, row
        price = float(price_str.replace(",", "."))
        assert price_unit == "Euro", row
        co2 = float(co2_str.replace(",", "."))
        assert co2_unit == "kg CO2", row
        return Row(date_str, Values(usage, price, co2))

    def parse(self, row: Row) -> Node:
        start = self.parse_date(row.date)
        end = self.calc_end(start)
        return Node(start, end, row.values, self.ts)

    def parse_date(self, date: str) -> datetime:
        raise NotImplementedError

    def calc_end(self, date: datetime) -> datetime:
        raise NotImplementedError


class Day(Parser):
    """
    One day with 4 values per hour.

    >>> Day(datetime(2025, 4, 30, 10, 16)).parse_date("jetzt")
    datetime.datetime(2025, 4, 30, 10, 15)
    >>> Day(datetime(2025, 4, 30, 10, 15)).parse_date("10:00")
    datetime.datetime(2025, 4, 30, 10, 0)
    >>> Day(datetime(2025, 1, 1, 0, 0)).parse_date("23:45")
    datetime.datetime(2024, 12, 31, 23, 45)
    """

    KEY = "24h"
    SPAN = "24 Stunden"

    def parse_date(self, date: str) -> datetime:
        if date == "jetzt":
            minute = self.last.minute
            minute -= minute % 15  # previous quater
            return self.last.replace(minute=minute)
        h_str, _, m_str = date.partition(":")
        assert m_str, date
        h = int(h_str)
        m = int(m_str)
        now = self.last.replace(hour=h, minute=m)
        if self.last < now:
            now -= timedelta(days=1)
        assert now < self.last, f"{now=} {self.last=}"
        self.last = now
        return now

    def calc_end(self, date: datetime) -> datetime:
        return date + timedelta(minutes=15)


class Week(Parser):
    """
    One week with 4 values per day.

    >>> Week(datetime(2025, 4, 30)).parse_date("Di.")
    datetime.datetime(2025, 4, 29, 0, 0)
    >>> Week(datetime(2025, 1, 1)).parse_date("Di.")
    datetime.datetime(2024, 12, 31, 0, 0)
    """

    KEY = "week"
    SPAN = "1 Woche"

    WD: ClassVar[dict[str, int]] = {
        "Mo.": 0,
        "Di.": 1,
        "Mi.": 2,
        "Do.": 3,
        "Fr.": 4,
        "Sa.": 5,
        "So.": 6,
    }

    def parse_date(self, date: str) -> datetime:
        try:
            h = int(date)
            assert h in {6, 12, 18}, date
        except ValueError:
            # wd = self.WD[date]
            h = 0
        now = self.last.replace(hour=h, minute=0)
        if self.last <= now:
            now -= timedelta(days=1)
            # assert now.weekday() == wd
        assert now < self.last, f"{now=} {self.last=}"
        self.last = now
        return now

    def calc_end(self, date: datetime) -> datetime:
        return date + timedelta(hours=6)


class Month(Parser):
    """
    One month with 1 value per day.

    >>> Month(datetime(2025, 4, 30)).parse_date("30.4.")
    datetime.datetime(2025, 5, 1, 0, 0)
    >>> Month(datetime(2025, 4, 1)).parse_date("31.3.")
    datetime.datetime(2025, 4, 1, 0, 0)
    >>> Month(datetime(2025, 1, 1)).parse_date("31.12.")
    datetime.datetime(2025, 1, 1, 0, 0)
    """

    KEY = "month"
    SPAN = "1 Monat"

    def parse_date(self, date: str) -> datetime:
        d_str, _, m_str = date.removesuffix(".").partition(".")
        assert m_str, date
        d = int(d_str)
        m = int(m_str)
        now = self.last.replace(month=m, day=d, hour=0, minute=0)  # end-of-day
        if self.last < now:
            now = now.replace(year=now.year - 1)
        assert now <= self.last, f"{now=} {self.last=}"
        now += timedelta(days=1)
        self.last = now
        return now

    def calc_end(self, date: datetime) -> datetime:
        return date + timedelta(days=1)


class Year(Parser):
    """
    2 years with 1 value per month.

    >>> Year(datetime(2025, 4, 30)).parse_date("April 2025")
    datetime.datetime(2025, 5, 1, 0, 0)
    >>> Year(datetime(2025, 1, 31)).parse_date("Januar 2025")
    datetime.datetime(2025, 2, 1, 0, 0)
    >>> Year(datetime(2025, 1, 1)).parse_date("Dezember 2024")
    datetime.datetime(2025, 1, 1, 0, 0)
    """

    KEY = "2years"
    SPAN = "2 Jahre"

    MONTH: ClassVar[dict[str, int]] = {
        "Januar": 1,
        "Februar": 2,
        "März": 3,
        "April": 4,
        "Mai": 5,
        "Juni": 6,
        "Juli": 7,
        "August": 8,
        "September": 9,
        "Oktober": 10,
        "November": 11,
        "Dezember": 12,
    }

    def parse_date(self, date: str) -> datetime:
        m_str, _, y_str = date.partition(" ")
        assert y_str, date
        m = self.MONTH[m_str] % 12 + 1  # next month 1st
        y = int(y_str) + (m == 1)
        now = self.last.replace(year=y, month=m, day=1, hour=0, minute=0)
        self.last = now
        return now

    def calc_end(self, date: datetime) -> datetime:
        return (date + timedelta(days=31)).replace(day=1)


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--output", "-o",
        type=FileType("w"),
        help="Write accumulated data",
    )
    parser.add_argument(
        "--raw", "-r",
        action="store_true",
        help="Output raw values",
    )
    if plt:
        parser.add_argument(
            "--plot", "-p",
            action="store_true",
            help="Plot values",
        )
    parser.add_argument(
        "files",
        type=Path,
        nargs="+",
        help="CSV files",
    )
    return parser.parse_args()


def parse_csv(args: Namespace) -> IntervalTree:
    tree = IntervalTree()

    for fn in args.files:
        parser = Parser.detect(fn)

        with fn.open() as fd:  # newline=""
            key, sep, val = fd.readline().strip().partition("=")
            delimiter = val if key == "sep" else ";"

            reader = csv.reader(fd, delimiter=delimiter)
            it = iter(reader)
            try:
                parser.parse_header(next(it))
                rows = [parser.parse_row(row) for row in it]
            except csv.Error as exc:
                sys.exit(f"file {fn}, line {reader.line_num}: {exc}")

            for row in reversed(rows):
                rec = parser.parse(row)
                if args.verbose:
                    print(rec)
                tree[rec.start:rec.end] = rec  # type: ignore[misc]

    return tree


def reduce(args: Namespace, tree: IntervalTree) -> None:
    if args.verbose:
        print(len(tree), file=sys.stderr)

    def drop_older(a: Node, b: Node) -> Node:
        return a if a.ts > b.ts else b

    tree.merge_equals(data_reducer=drop_older)

    if args.verbose:
        print(len(tree), file=sys.stderr)

    tree.split_overlaps()

    if args.verbose:
        print(len(tree), file=sys.stderr)

    def sub_specific(a: Node, b: Node) -> Node:
        if a.delta < b.delta:
            child, parent = a, b
        elif b.delta < a.delta:
            child, parent = b, a
        else:
            raise AssertionError(a, b)

        parent -= child
        return child

    tree.merge_equals(data_reducer=sub_specific)

    if args.verbose:
        print(len(tree), file=sys.stderr)


def main() -> None:
    args = parse_args()
    data = parse_csv(args)
    reduce(args, data)

    total = 0.0
    dates, values = [], []
    for rec in sorted(data):
        start, end, data = rec
        span = end - start
        h, m = divmod(span.seconds // 60, 60)

        usage = data.values.usage * (span / data.span)
        total += usage
        if args.raw:
            print(f"{start:%Y-%m-%d %H:%M}\t{span.days:2}d{h:2}h{m:2}m\t{usage:10.0f}")
        if args.output:
            args.output.write(f"{end:%Y-%m-%d %H:%M}\t{total:10.0f}\n")
        if args.plot:
            dates.append(end)
            values.append(total)

    if plt and args.plot:
        fig, ax = plt.subplots()
        ax.plot(dates, values)
        plt.show()


if __name__ == "__main__":
    main()
