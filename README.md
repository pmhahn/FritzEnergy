# AVM FRITZ!Smart Energy 200

This reads CVS files exported by FRITZ!Box, parses them and outputs sanitized data.
There are 4 different formats spanning different time spans:
- One day with 4 values per hour
- One week with 4 values per day
- One month with 1 value per day
- Two years with 1 value per month

Different files may have data for the same timespan.
In that case the timestamp of generation is consulted to prefer later data.

Intervals may also overlap.
In that case the larger interval is split into multiple smaller intervals and adjusted by the more specific interval.

## Usage

Display CSV data as plot:

```console
$ uv tool install -e .[plt]
$ fritz-strom --plot csv/*.csv
```

As an alternative use `… --output ./output.csv` instead to write sanitized data into one CSV file, which then can be used by other programs like LibreCalc.

## Links
- https://github.com/chaimleib/intervaltree
- https://pmhahn.github.io/avm-smart-energy-200-csv/
