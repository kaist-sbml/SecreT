# SecreT
Secretion Transformer
<<<<<<< HEAD
* Note: This source code was developed in Linux, and has been tested in Ubuntu 16.04 with Python 3.6 and CUDA version 10.1

## Source

1. Clone the repository
```
git clone https://github.com/kaist-sbml/SecreT.git
```

2. Create and activate a conoda environment
```
conda env create -f environment.yml
conda activate SecreT
```

## Example
* Run SecreT
```
python run_SecreT.py --input input_example.txt --type Sec --outdir output_example
```

--input: Input txt file (identifier, sequence per line) 

--type: Signal peptide type (one of Sec or Tat)

--temperature: Temperature for decoding. (default: Sec→0.5, Tat→1.5)

--num-beams: Beam width (default: 100)

--outdir: Output directory basename (default: output)

