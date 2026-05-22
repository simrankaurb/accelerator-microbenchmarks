"""Utility to load benchmark parameters from CSV files."""

import csv
import io
from typing import Any
import urllib.error
import urllib.request


def load_shapes_from_csv(path: str) -> list[dict[str, Any]]:
  """Reads a CSV (local or remote URL) and returns a list of row dicts.

  Supports Google Sheets export links:
  https://docs.google.com/spreadsheets/d/<ID>/export?format=csv&gid=<GID>

  Args:
    path: The path to the CSV file, can be a local path or a URL.

  Returns:
    A list of dictionaries, where each dictionary represents a row in the CSV.
    Column headers are keys and inferred-type values are values.
  """
  if not path:
    return []

  shapes = []
  try:
    if path.startswith(('http://', 'https://')):
      with urllib.request.urlopen(path) as response:
        content = response.read().decode('utf-8')
        f = io.StringIO(content)
    else:
      f = open(path, mode='r', encoding='utf-8')

    with f:
      reader = csv.DictReader(f)
      for row in reader:
        # Infer types
        typed_row = {}
        for k, v in row.items():
          if v is None or not v:
            typed_row[k] = None
            continue

          # Try int
          try:
            typed_row[k] = int(v)
            continue
          except ValueError:
            pass

          # Try float
          try:
            typed_row[k] = float(v)
            continue
          except ValueError:
            pass

          # Stay as string
          typed_row[k] = v
        shapes.append(typed_row)
  except (urllib.error.URLError, OSError, UnicodeDecodeError, csv.Error) as e:
    print(f'Error loading CSV from {path}: {e}')

  return shapes
