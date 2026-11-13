"""Custom Streaming CSV Reader.

Provides chunked reading and feature/label separation for streaming classification
pipelines using pure Python and NumPy (without pandas).
"""
from __future__ import annotations

import itertools
import os
from typing import Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np

from numcompute.io import MISSING_TOKENS, read_header

__all__ = ["CSVReader"]


class CSVReader:
    """Streamed CSV reader supporting chunk iteration and target separation.

    Parameters
    ----------
    path : str or PathLike
        Path to CSV file.
    target_column : int or str or None, optional
        Target column index or name (if header exists). If None, all columns
        are treated as features. Default None.
    delimiter : str, optional
        Field separator. Default ','.
    skip_header : int, optional
        Number of header rows to skip. Default 1.
    has_header : bool, optional
        Whether the first non-comment line is a header. Default True.
    comments : str, optional
        Comment prefix character. Default '#'.
    encoding : str, optional
        Text encoding. Default 'utf-8'.
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        target_column: Optional[Union[int, str]] = None,
        delimiter: str = ",",
        skip_header: int = 1,
        has_header: bool = True,
        comments: str = "#",
        encoding: str = "utf-8",
    ) -> None:
        self.path = os.fspath(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"CSV file not found: {self.path!r}")

        if not isinstance(delimiter, str) or len(delimiter) == 0:
            raise ValueError(f"delimiter must be a non-empty string, got {delimiter!r}")

        self.target_column = target_column
        self.delimiter = delimiter
        self.skip_header = skip_header
        self.has_header = has_header
        self.comments = comments
        self.encoding = encoding

        self._headers: Optional[List[str]] = None
        self._target_idx: Optional[int] = None
        self._feature_indices: Optional[List[int]] = None

    def get_column_names(self) -> List[str]:
        """Read column names from CSV header."""
        if self._headers is None:
            if self.has_header:
                self._headers = read_header(
                    self.path, delimiter=self.delimiter, encoding=self.encoding
                )
            else:
                # Infer number of columns from first data line
                with open(self.path, "r", encoding=self.encoding) as fh:
                    for line in fh:
                        s = line.strip()
                        if s and not s.startswith(self.comments):
                            cols = [c.strip() for c in s.split(self.delimiter)]
                            self._headers = [f"col_{i}" for i in range(len(cols))]
                            break
                if self._headers is None:
                    raise ValueError(f"CSV file {self.path!r} is empty")
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

    def _parse_lines(self, lines: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Parse a list of raw CSV lines into (X_features, y_target)."""
        if not lines:
            raise ValueError("No lines to parse")

        rows = []
        for line in lines:
            row = [item.strip() for item in line.rstrip("\r\n").split(self.delimiter)]
            rows.append(row)

        n_cols = len(rows[0])
        for r in rows:
            if len(r) != n_cols:
                raise ValueError(
                    f"Ragged CSV row detected: expected {n_cols} columns, got {len(r)}"
                )

        target_idx = self._resolve_target_index(n_cols)
        raw_arr = np.array(rows, dtype=str)

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
            # Replace missing tokens
            col_floats = []
            for val in col:
                if val in missing_set:
                    col_floats.append(np.nan)
                else:
                    try:
                        col_floats.append(float(val))
                    except ValueError:
                        raise ValueError(f"Could not convert field {val!r} to float in column {j}")
            X_float[:, j] = col_floats

        # Process target vector if present
        y_out: Optional[np.ndarray] = None
        if raw_y is not None:
            # Check if target is numeric int or float
            try:
                # Try int first if all values are integer strings
                y_float = raw_y.astype(float)
                if np.all(np.equal(np.mod(y_float, 1), 0)):
                    y_out = y_float.astype(int)
                else:
                    y_out = y_float
            except ValueError:
                # Keep as string labels
                y_out = raw_y

        return X_float, y_out

    def read(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Read entire CSV file into (X, y) tuple."""
        lines: List[str] = []
        with open(self.path, "r", encoding=self.encoding) as fh:
            for _ in range(self.skip_header):
                next(fh, None)
            for line in fh:
                s = line.strip()
                if s and not s.startswith(self.comments):
                    lines.append(line)

        if not lines:
            raise ValueError(f"No data rows found in {self.path!r}")

        return self._parse_lines(lines)

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
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

        with open(self.path, "r", encoding=self.encoding) as fh:
            for _ in range(self.skip_header):
                next(fh, None)

            def _valid_lines():
                for line in fh:
                    s = line.strip()
                    if s and not s.startswith(self.comments):
                        yield line

            line_gen = _valid_lines()
            while True:
                block = list(itertools.islice(line_gen, chunk_size))
                if not block:
                    break
                yield self._parse_lines(block)
