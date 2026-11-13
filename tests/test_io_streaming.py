"""Tests for CSVReader streaming I/O."""
from __future__ import annotations

import os
import numpy as np
import pytest

from numcompute_stream.io import CSVReader


def test_csv_reader_full_read_with_header(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "feat1,feat2,label\n"
        "1.0,2.0,0\n"
        "3.0,4.0,1\n"
        "5.0,6.0,0\n"
    )

    reader = CSVReader(csv_file, target_column="label")
    assert reader.get_column_names() == ["feat1", "feat2", "label"]

    X, y = reader.read()
    assert X.shape == (3, 2)
    assert y is not None
    assert y.shape == (3,)
    np.testing.assert_allclose(X[0], [1.0, 2.0])
    np.testing.assert_array_equal(y, [0, 1, 0])


def test_csv_reader_quoted_fields(tmp_path):
    # CSV with quoted text containing commas
    csv_file = tmp_path / "quoted.csv"
    csv_file.write_text(
        "val1,val2,category\n"
        '1.5,2.5,"Class, Alpha"\n'
        '3.5,4.5,"Class, Beta"\n'
    )

    reader = CSVReader(csv_file, target_column="category")
    X, y = reader.read()
    assert X.shape == (2, 2)
    assert y is not None
    np.testing.assert_array_equal(y, ["Class, Alpha", "Class, Beta"])


def test_csv_reader_chunk_iteration_matches_full_read(tmp_path):
    csv_file = tmp_path / "chunks.csv"
    lines = ["f1,f2,target\n"] + [f"{i}.0,{i*2}.0,{i%2}\n" for i in range(11)]
    csv_file.write_text("".join(lines))

    reader = CSVReader(csv_file, target_column="target")
    full_X, full_y = reader.read()

    # Iterate in chunks of 4 (4, 4, 3)
    chunks = list(reader.iter_chunks(chunk_size=4))
    assert len(chunks) == 3
    assert len(chunks[0][0]) == 4
    assert len(chunks[1][0]) == 4
    assert len(chunks[2][0]) == 3

    concat_X = np.vstack([c[0] for c in chunks])
    concat_y = np.concatenate([c[1] for c in chunks])

    np.testing.assert_allclose(concat_X, full_X)
    np.testing.assert_array_equal(concat_y, full_y)


def test_csv_reader_no_header_mode(tmp_path):
    csv_file = tmp_path / "no_header.csv"
    csv_file.write_text(
        "10.0,20.0,1\n"
        "30.0,40.0,2\n"
    )

    reader = CSVReader(csv_file, target_column=-1, has_header=False)
    assert reader.get_column_names() == ["col_0", "col_1", "col_2"]

    X, y = reader.read()
    assert X.shape == (2, 2)
    np.testing.assert_allclose(X[0], [10.0, 20.0])
    np.testing.assert_array_equal(y, [1, 2])


def test_csv_reader_comments_and_blank_lines(tmp_path):
    csv_file = tmp_path / "comments.csv"
    csv_file.write_text(
        "# Metadata comment\n"
        "\n"
        "a,b,target\n"
        "# Mid-data comment\n"
        "1.0,2.0,0\n"
        "3.0,4.0,1\n"
    )

    reader = CSVReader(csv_file, target_column="target")
    X, y = reader.read()
    assert len(X) == 2


def test_csv_reader_missing_values_as_nan(tmp_path):
    csv_file = tmp_path / "missing.csv"
    csv_file.write_text(
        "a,b,target\n"
        "1.0,NA,0\n"
        "?,4.0,1\n"
        "5.0,NaN,0\n"
    )

    reader = CSVReader(csv_file, target_column="target")
    X, y = reader.read()

    assert np.isnan(X[0, 1])
    assert np.isnan(X[1, 0])
    assert np.isnan(X[2, 1])
    assert not np.isnan(X[0, 0])
    assert not np.isnan(X[1, 1])


def test_csv_reader_custom_delimiter(tmp_path):
    tsv_file = tmp_path / "data.tsv"
    tsv_file.write_text(
        "x\ty\tlabel\n"
        "1.5\t2.5\tA\n"
        "3.5\t4.5\tB\n"
    )

    reader = CSVReader(tsv_file, target_column="label", delimiter="\t")
    X, y = reader.read()
    assert X.shape == (2, 2)
    assert y is not None
    np.testing.assert_array_equal(y, ["A", "B"])


def test_csv_reader_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        CSVReader(tmp_path / "non_existent.csv")

    dummy = tmp_path / "dummy.csv"
    dummy.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="delimiter"):
        CSVReader(dummy, delimiter="")

    with pytest.raises(ValueError, match="skip_header"):
        CSVReader(dummy, skip_header=-1)

    with pytest.raises(ValueError, match="skip_header"):
        CSVReader(dummy, skip_header=True)

    with pytest.raises(ValueError, match="chunk_size"):
        list(CSVReader(dummy).iter_chunks(chunk_size=0))

    with pytest.raises(ValueError, match="chunk_size"):
        list(CSVReader(dummy).iter_chunks(chunk_size=True))

    reader = CSVReader(dummy, target_column="missing_col")
    with pytest.raises(ValueError, match="not found in headers"):
        reader.read()

    ragged = tmp_path / "ragged.csv"
    ragged.write_text("a,b\n1,2\n1,2,3\n")
    with pytest.raises(ValueError, match="Ragged CSV row"):
        CSVReader(ragged).read()
