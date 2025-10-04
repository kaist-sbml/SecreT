# SecreT
Secretion Transformer

* Note: This source code was developed in Linux, and has been tested in Ubuntu 16.04 with Python 3.6 and CUDA version 10.1

## Source

### 1) Clone the repository
```
git clone https://github.com/kaist-sbml/SecreT.git
```

### 2) Create and activate a conda environment
```
conda env create -f environment.yml
conda activate SecreT
pip install -r requirements.txt
```

**Note about PyTorch installation failures**

If installation fails (often due to CUDA/driver mismatch), please install PyTorch that matches your setup.

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

## Data preparation & preprocessing

This project supports building an SP dataset from UniProt JSON exports and refining it with SignalP 6.0.

### 1) Download data
- **UniProt**: Download your protein records **in JSON format** (e.g., via UniProt website or API) and place them under a directory (e.g., `Uniprot_download/`).
- **SignalP 6.0**: Download from https://services.healthtech.dtu.dk/services/SignalP-6.0/ and follow their installation instructions.

### 2) Build SP dataset from UniProt JSON
Extract signal peptide (SP) and mature protein sequences from UniProt JSON files and save as a single pickle:
```
python build_sp_dataset.py --input-dir Uniprot_download --output SP_dataset.pkl
```

### 3) Filter with SignalP 6.0 and split SP regions

> **Terminology note (to avoid confusion):**  
> In our manuscript, **“SP”** is used as an abbreviation for the generic term **signal peptide**.  
> In **SignalP 6.0**, however, the label **`SP`** means **Sec/SPI** specifically. Other secretion types are labeled as:
> - **Sec/SPII** → `LIPO`
> - **Tat/SPI** → `TAT`
> - **Tat/SPII** → `TATLIPO`
> - **Sec/SPIII** → `PILIN`

We keep only **Sec/SPI (`SP`)** and **Tat/SPI (`TAT`)** where the predicted cleavage site matches the UniProt-derived SP length, and then split the SP into regions:

```
python preprocess_signalp_regions.py --input SP_dataset.pkl --output SP_dataset_processed.pkl
```

Keep only Sec/SPI (SP) and Tat/SPI (TAT) where the predicted cleavage site matches the UniProt-derived SP length, and split SP into regions:
```
python preprocess_signalp_regions.py --input SP_dataset.pkl --output SP_dataset_processed.pkl
```
Output is a pickle list with tuples of:

SP: (identifier, "SP", (n, h, c), mature_sequence)

TAT: (identifier, "TAT", (n, RR, h, c), mature_sequence)



