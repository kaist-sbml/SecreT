import argparse
import glob
import os
import pickle
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from signalp6h_fast.signalp.predict import predict
from signalp6h_fast.signalp.utils import tokenize_sequence, get_cleavage_sites

# Paths
BASE_DIR = "./signalp6h_fast/signalp/"
FAST_MODEL_PATH = os.path.join(BASE_DIR, "model_weights/distilled_model_signalp6_gpu.pt")

# Label maps
GLOBAL_LABEL_DICT = {0: "OTHER", 1: "SP", 2: "LIPO", 3: "TAT", 4: "TATLIPO", 5: "PILIN"}


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter Sec/SPI and Tat/SPI with SignalP6h Fast and split regions."
    )
    p.add_argument("--input", required=True, help="Input .pkl file or directory containing .pkl files.")
    p.add_argument("--output", required=True, help="Output .pkl path for the processed dataset.")
    return p.parse_args(list(argv) if argv is not None else None)


def list_input_pkls(path: str) -> List[str]:
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.pkl")))
    if os.path.isfile(path) and path.lower().endswith(".pkl"):
        return [path]
    raise FileNotFoundError("Input must be a .pkl file or a directory: {}".format(path))


def model_inputs_from_sequences(
    sequences: List[Tuple[str, str, str]], kingdom_id: str = "other"
) -> Tuple[List[str], List[str], List[str], torch.LongTensor, torch.LongTensor]:
    """Pad/truncate per SignalP6h practice; keep length 73 with mask. (Fixed shape expected by model)"""
    identifiers = [t[0] for t in sequences]
    sps = [t[1] for t in sequences]
    ps = [t[2] for t in sequences]
    seqs = [s + p for s, p in zip(sps, ps)]

    toks = [tokenize_sequence(x[:70], kingdom_id) for x in seqs]
    toks = [ids + [0] * (73 - len(ids)) for ids in toks]
    arr = np.vstack(toks)
    mask = (arr > 0).astype(np.int64)

    return identifiers, sps, ps, torch.LongTensor(arr), torch.LongTensor(mask)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    inputs = list_input_pkls(args.input)

    # Load once
    model = torch.jit.load(FAST_MODEL_PATH)
    model.eval()

    new_dataset: List[Tuple[str, str, Tuple[str, ...], str]] = []
    sec_count = 0
    tat_count = 0
    error_tuples: List[Tuple[str, int]] = []

    for pkl_path in inputs:
        with open(pkl_path, "rb") as f:
            sp_dataset: List[Tuple[str, str, str]] = pickle.load(f)

        identifiers, sps, ps, input_ids, input_mask = model_inputs_from_sequences(sp_dataset, "other")
        global_probs, _, viterbi_paths = predict(model, input_ids, input_mask, batch_size=1)
        pred_label_id = np.argmax(global_probs, axis=1)
        cleavage_sites = get_cleavage_sites(viterbi_paths)

        for idx, identifier in tqdm(
            list(enumerate(identifiers)), desc=os.path.basename(pkl_path), unit="rec"
        ):
            try:
                if any(x in sps[idx] or x in ps[idx] for x in ("B", "U", "X", "Z")):
                    continue

                prediction = GLOBAL_LABEL_DICT[int(pred_label_id[idx])]

                if prediction == "SP" and int(cleavage_sites[idx]) == len(sps[idx]):
                    v = viterbi_paths[idx]
                    flag1 = np.argwhere(v == 4)[0].item()
                    flag2 = np.argwhere(v == 5)[0].item()
                    n, h, c = sps[idx][:flag1], sps[idx][flag1:flag2], sps[idx][flag2:]
                    new_dataset.append((identifier, prediction, (n, h, c), ps[idx]))
                    sec_count += 1
                    continue

                if prediction == "TAT" and int(cleavage_sites[idx]) == len(sps[idx]):
                    v = viterbi_paths[idx]
                    flag1 = np.argwhere(v == 17)[0].item()
                    flag2 = np.argwhere(v == 18)[0].item()
                    flag3 = np.argwhere(v == 19)[0].item()
                    n, rr = sps[idx][:flag1], sps[idx][flag1:flag2]
                    h, c = sps[idx][flag2:flag3], sps[idx][flag3:]
                    new_dataset.append((identifier, prediction, (n, rr, h, c), ps[idx]))
                    tat_count += 1
                    continue

            except Exception:
                error_tuples.append((os.path.basename(pkl_path), idx))

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(new_dataset, f, pickle.HIGHEST_PROTOCOL)

    print("Processed files:", len(inputs))
    print("Kept -> Sec/SPI: {}, Tat/SPI: {}, Errors: {}".format(sec_count, tat_count, len(error_tuples)))
    print("Output: {} | Total records: {}".format(out_path, len(new_dataset)))
    if error_tuples:
        print("Example errors (file, idx):", error_tuples[:5])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
