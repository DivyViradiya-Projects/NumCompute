"""Custom Streaming CSV Reader.

Provides chunked reading and feature/label separation for streaming classification
pipelines using pure Python standard library and NumPy (without pandas).
"""
from __future__ import annotations

import csv
import itertools
import os
from typing import Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np

from numcompute.io import MISSING_TOKENS, read_header

__all__ = ["CSVReader"]


class CSVReader:
    """Streamed CSV reader supporting chunk iteration and target separation.

    Uses standard Python ``csv.reader`` for robust RFC-4180 parsing (handles quoted
    fields, escaped delimiters, and whitespace) while maintaining a minimal memory footprint.

    Parameters
    ----------
    path : str or PathLike
        Path to CSV file.
    target_column : int or str or None, optional
        Target column index or header name. If None, all columns are returned as features.
    delimiter : str, optional
        Field separator. Default ','.
    has_header : bool, optional
        Whether the first non-comment line contains column names. Default True.
    skip_header : int, optional
        Number of physical rows to skip at the start of the file before header/data. Default 0.
    comments : str, optional
        Comment prefix character. Lines starting with this are ignored. Default '#'.
    encoding : str, optional
        Text encoding. Default 'utf-8'.
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        target_column: Optional[Union[int, str]] = None,
        delimiter: str = ",",
        has_header: bool = True,
        skip_header: int = 0,
        comments: str = "#",
        encoding: str = "utf-8",
    ) -> None:
        self.path = os.fspath(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"CSV file not found: {self.path!r}")

        if not isinstance(delimiter, str) or len(delimiter) == 0:
            raise ValueError(f"delimiter must be a non-empty string, got {delimiter!r}")

        if (
            isinstance(skip_header, bool)
            or not isinstance(skip_header, (int, np.integer))
            or skip_header < 0
        ):
            raise ValueError(f"skip_header must be a non-negative int, got {skip_header!r}")

        self.target_column = target_column
        self.delimiter = delimiter
        self.has_header = has_header
        self.skip_header = int(skip_header)
        self.comments = comments
        self.encoding = encoding

        self._headers: Optional[List[str]] = None

    def get_column_names(self) -> List[str]:
        """Read or infer column names."""
        if self._headers is None:
            with open(self.path, "r", encoding=self.encoding) as fh:
                for _ in range(self.skip_header):
                    next(fh, None)
                rdr = csv.reader(fh, delimiter=self.delimiter)
                for row in rdr:
                    if row and not row[0].startswith(self.comments):
                        if self.has_header:
                            self._headers = [c.strip() for c in row]
                        else:
                            self._headers = [f"col_{i}" for i in range(len(row))]
                        break
            if self._headers is None:
                raise ValueError(f"CSV file {self.path!r} is empty or contains only comments")
        return list(self._headers)

    def _resolve_target_index(self, n_cols: int) -> Optional[int]:
        if self.target_column is None:
            return None
        if isinstance(self.target_column, str):
            names = self.get_column_names()
            if self.target_column not in names:
                raise ValueError(
                    f"target_column {self.target_column!r} not found in headers: {names}"
                )
            return names.index(self.target_column)
        elif isinstance(self.target_column, (int, np.integer)):
            idx = int(self.target_column)
            if not (-n_cols <= idx < n_cols):
                raise ValueError(
                    f"target_column index {idx} out of range for {n_cols} columns"
                )
            return idx % n_cols
        else:
            raise TypeError(
                f"target_column must be str, int, or None, got {type(self.target_column).__name__}"
            )

    def _parse_rows(self, raw_rows: List[List[str]]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Parse parsed CSV row strings into (X_features, y_target)."""
        if not raw_rows:
            raise ValueError("No data rows to parse")

        n_cols = len(raw_rows[0])
        for idx, r in enumerate(raw_rows):
            if len(r) != n_cols:
                raise ValueError(
                    f"Ragged CSV row detected at row {idx}: expected {n_cols} columns, got {len(r)}"
                )

        target_idx = self._resolve_target_index(n_cols)
        raw_arr = np.array(raw_rows, dtype=str)

        if target_idx is not None:
            feature_indices = [i for i in range(n_cols) if i != target_idx]
            raw_X = raw_arr[:, feature_indices]
            raw_y = raw_arr[:, target_idx]
        else:
            raw_X = raw_arr
            raw_y = None

        # Convert feature matrix to float, replacing missing tokens with NaN
        X_float = np.empty(raw_X.shape, dtype=float)
        missing_set = set(MISSING_TOKENS)
        for j in range(raw_X.shape[1]):
            col = raw_X[:, j]
            col_floats = []
            for val in col:
                val_stripped = val.strip()
                if val_stripped in missing_set:
                    col_floats.append(np.nan)
                else:
                    try:
                        col_floats.append(float(val_stripped))
                    except ValueError:
                        raise ValueError(
                            f"Could not convert field {val!r} to float in feature column {j}"
                        )
            X_float[:, j] = col_floats

        # Process target vector if present
        y_out: Optional[np.ndarray] = None
        if raw_y is not None:
            try:
                y_float = raw_y.astype(float)
                if np.all(np.equal(np.mod(y_float, 1), 0)):
                    y_out = y_float.astype(int)
                else:
                    y_out = y_float
            except ValueError:
                # Retain as string labels
                y_out = raw_y

        return X_float, y_out

    def _iter_valid_rows(self, fh) -> Iterator[List[str]]:
        """Yield non-comment, non-empty parsed CSV rows after headers/skips."""
        for _ in range(self.skip_header):
            next(fh, None)

        rdr = csv.reader(fh, delimiter=self.delimiter)
        header_skipped = not self.has_header

        for row in rdr:
            if not row or (row[0].startswith(self.comments)):
                continue
            if not header_skipped:
                header_skipped = True
                continue
            yield [c.strip() for c in row]

    def read(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Read entire CSV file into (X, y) tuple."""
        with open(self.path, "r", encoding=self.encoding) as fh:
            rows = list(self._iter_valid_rows(fh))

        if not rows:
            raise ValueError(f"No data rows found in {self.path!r}")

        return self._parse_rows(rows)

    def iter_chunks(
        self, chunk_size: int = 100
    ) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Iterate over dataset in chunks of at most chunk_size rows.

        Parameters
        ----------
        chunk_size : int, optional
            Number of rows per chunk (>= 1). Default 100.

        Yields
        ------
        (X_chunk, y_chunk) : Tuple[ndarray, ndarray or None]
            Feature chunk and target chunk.
        """
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
            raise ValueError(f"chunk_size must be an integer >= 1, got {chunk_size!r}")

        with open(self.path, "r", encoding=self.encoding) as fh:
            row_gen = self._iter_valid_rows(fh)
            while True:
                block = list(itertools.islice(row_gen, int(chunk_size)))
                if not block:
                    break
                yield self._parse_rows(block)
