"""Custom Streaming CSV Reader.

Provides chunked reading and feature/label separation for streaming classification
pipelines using pure Python and NumPy (without pandas).
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np

from numcompute.io import iter_csv_chunks, load_csv, read_header

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
        Number of header rows to skip. Default 0.
    has_header : bool, optional
        Whether the file has a header line. Default True.
    """

    def __init__(
        self,
        path,
        target_column: Optional[Union[int, str]] = None,
        delimiter: str = ",",
        skip_header: int = 1,
        has_header: bool = True,
    ) -> None:
        self.path = path
        self.target_column = target_column
        self.delimiter = delimiter
        self.skip_header = skip_header
        self.has_header = has_header

    def get_column_names(self) -> List[str]:
        """Read column names from CSV header."""
        raise NotImplementedError("Will be implemented in Phase 5")

    def read(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Read entire CSV into (X, y)."""
        raise NotImplementedError("Will be implemented in Phase 5")

    def iter_chunks(
        self, chunk_size: int = 100
    ) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Iterate over dataset in chunks of at most chunk_size rows."""
        raise NotImplementedError("Will be implemented in Phase 5")
