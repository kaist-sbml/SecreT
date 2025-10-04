import argparse
import json
import logging
import os
import re
import sys
import glob
import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

Accession = str
Sequence = str
SPTuple = Tuple[Accession, Sequence, Sequence]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract signal peptide and following sequences from UniProt JSON files."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing UniProt JSON files (*.json).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output pickle file (e.g., ./SP_dataset.pkl).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def setup_logging() -> None:
    # Why: default INFO-level visibility is useful for batch runs.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def safe_leading_int(value: Any, default: int = 9) -> int:
    s = str(value)
    m = re.match(r"\s*(\d+)", s)
    return int(m.group(1)) if m else default


def find_signal_span(features: List[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    for feat in features or []:
        if feat.get("type") == "Signal":
            try:
                start = int(feat["location"]["start"]["value"])
                end = int(feat["location"]["end"]["value"])
                if start >= 1 and end >= start:
                    return start, end
            except Exception:
                continue
    return None


def extract_from_result(protein: Dict[str, Any]) -> Optional[SPTuple]:
    pe_raw = protein.get("proteinExistence")
    pe_field = pe_raw[0] if isinstance(pe_raw, list) and pe_raw else pe_raw
    pe = safe_leading_int(pe_field, default=9)
    if pe >= 4:
        return None

    seq: str = (protein.get("sequence") or {}).get("value") or ""
    if not seq:
        return None

    span = find_signal_span(protein.get("features") or [])
    if not span:
        return None

    start, end = span
    if end > len(seq):
        return None

    sp = seq[start - 1 : end]
    post = seq[end:]
    accession = protein.get("primaryAccession") or protein.get("uniProtkbId") or ""
    if not accession:
        return None

    return accession, sp, post


def read_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        logging.error("Input directory does not exist: %s", input_dir)
        return 2

    pattern = os.path.join(input_dir, "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        logging.error("No JSON files found under: %s", pattern)
        return 2

    logging.info("Found %d JSON file(s).", len(files))

    sp_p_list: List[SPTuple] = []
    total_succeed = total_remove = total_fail = 0

    for fp in files:
        logging.info("Reading: %s", fp)
        try:
            data = read_json_file(fp)
        except Exception as e:
            logging.warning("Failed to read %s: %s", fp, e)
            continue

        results = data.get("results") or []
        succeed_ids: List[str] = []
        remove_ids: List[str] = []
        fail_ids: List[str] = []

        for protein in tqdm(results, desc=os.path.basename(fp), unit="rec"):
            try:
                tup = extract_from_result(protein)
                if tup is None:
                    remove_ids.append(protein.get("primaryAccession", "NA"))
                    continue
                sp_p_list.append(tup)
                succeed_ids.append(tup[0])
            except Exception:
                fail_ids.append(protein.get("primaryAccession", "NA"))

        logging.info(
            "File summary -> succeed: %d, remove: %d, fail: %d",
            len(succeed_ids),
            len(remove_ids),
            len(fail_ids),
        )
        total_succeed += len(succeed_ids)
        total_remove += len(remove_ids)
        total_fail += len(fail_ids)
        data = None

    logging.info(
        "Accumulated -> succeed: %d, remove: %d, fail: %d",
        total_succeed,
        total_remove,
        total_fail,
    )

    if not sp_p_list:
        logging.error("No SP entries extracted. Nothing to write.")
        return 1

    accs, sps, posts = zip(*sp_p_list)
    df = pd.DataFrame({"accession_id": accs, "sp": sps, "p": posts})
    before = len(df)
    df = df.drop_duplicates(subset=["p"], keep=False)
    after = len(df)
    logging.info("Dropped duplicates by 'p': %d -> %d", before, after)

    final_list: List[SPTuple] = list(
        zip(df["accession_id"].tolist(), df["sp"].tolist(), df["p"].tolist())
    )
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    try:
        with open(out_path, "wb") as fh:
            pickle.dump(final_list, fh)
        logging.info("Wrote %d tuples to %s", len(final_list), out_path)
    except Exception as e:
        logging.error("Failed to write pickle: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())