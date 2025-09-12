import argparse
from pathlib import Path
import torch
from SecreT import SecreTransformer, beam_search_with_threshold
from utils import tokenize_P, untokenize, num_tokens

def prepare_outdir(base: Path) -> Path:
    """Create a unique output directory.
    If `base` exists, append an increasing integer suffix starting from 1.
    """
    base = Path(base)
    if not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        return base
    n = 1
    while True:
        cand = base.parent / f"{base.name}{n}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand
        n += 1


def main():
    parser = argparse.ArgumentParser(description="Signal Peptide Predictor")
    parser.add_argument("--input", type=Path, required=True, help="Input txt file (identifier, sequence per line)")
    parser.add_argument("--type", choices=["Sec", "Tat"], required=True, help="Signal peptide type")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for decoding. Omit to use Sec→0.5, Tat→1.5")
    parser.add_argument("--num-beams", type=int, default=100, help="Beam width (default 100)")
    parser.add_argument("--outdir", type=Path, default=Path("output"), help="Output directory basename")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    model = SecreTransformer(num_tokens=num_tokens, dim_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, dropout_p=0.1).to(device)
    model.load_state_dict(torch.load("./model/SecreT+.pt", map_location=device))

    sp_type = args.type
    temperature = args.temperature if args.temperature is not None else (0.5 if sp_type == "Sec" else 1.5)
    num_beams = args.num_beams

    output_dir = prepare_outdir(args.outdir)
    print(f"Output directory: {output_dir}")

    with open(args.input, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                identifier, seq = [p.strip() for p in raw.split(",", 1)]
            except ValueError:
                print(f"[WARN] Skipping malformed line {lineno}: {raw}", file=sys.stderr)
                continue
                
            print(identifier)    
            seq = seq[:100]
            tokenized_X = tokenize_P(seq)
            tokenized_X = torch.tensor(tokenized_X, device=device).view(1, -1)
            start_token = 0 if sp_type == "Sec" else 1
            predicted_sequences = beam_search_with_threshold(
                model,
                tokenized_X,
                start_token,
                num_beams,
                75,
                0.01,
                temperature,
                device,
            )
            output_file = output_dir / f"{identifier}_{sp_type}_{temperature}_{num_beams}.txt"
            with open(output_file, "w", encoding="utf-8") as out:
                for pred in predicted_sequences:
                    out.write(f"{untokenize(pred)}")
            print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
