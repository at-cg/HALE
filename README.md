## <a name="intro"></a>Introduction

HALE (**H**aplotype-**A**ware **L**ong-read **E**rror correction) is a haplotype-aware error correction tool designed for long reads. It has been primarily evaluated on PacBio HiFi data, with planned extensions to support ONT simplex reads in the future. 


## <a name="requirements"></a> Requirements
- Linux OS (tested on RHEL 8 and Ubuntu 22.04)
- [rustup](https://rustup.rs/) (Install using [rustup](https://rustup.rs/), the recommended way to get Rust)
- Python 3.1 or above (and conda) for data preprocessing

  ### <a name="sys-requirements"></a> System Requirements
  Make sure the following system packages are installed (Linux):
  ```bash
  sudo apt update
  sudo apt install build-essential autoconf libtool pkg-config
  ```



## <a name="started"></a>Try HALE on Small Test Data
The entire test workflow below will take about 20-30 minutes. Users can either run the commands one by one or copy the commands into an executable script.

```sh
# Install HALE 
git clone https://github.com/at-cg/HALE.git
cd HALE && RUSTFLAGS="-Ctarget-cpu=native" cargo build -q --release

# Create conda env
conda env create --file scripts/hale-env.yml
conda activate hale

mkdir -p test_run && cd test_run/

# download small test dataset
wget -O HG002.chr19_10M_12M.fastq.gz https://zenodo.org/records/14048797/files/HG002.chr19_10M_12M.fastq.gz?download=1

# Get all read ids in a seperate file
seqkit seq -ni HG002.chr19_10M_12M.fastq.gz > HG002.chr19_10M_12M.read_ids

# Run all-vs-all overlap
../scripts/create_batched_alignments.sh HG002.chr19_10M_12M.fastq.gz HG002.chr19_10M_12M.read_ids 8 batch_alignments

# Run hale correct
../target/release/hale correct --read-alns batch_alignments -t 8 HG002.chr19_10M_12M.fastq.gz HG002.chr19_10M_12M_corrected.fa

```
For large inputs, users are recommended to increase the thread count depending on the number of the cores available for use. HALE takes about 9 minutes and ~50 GB RAM using 64 threads on a multicore [Perlmutter CPU-based node](https://docs.nersc.gov/systems/perlmutter/architecture/) to process 60x HiFi chr9 HG002 human genome dataset.




## <a name="install"></a>Installation

1. Clone the repository:
```sh
git clone https://github.com/at-cg/HALE.git
```

2. Compile the source code:
```sh
cd HALE
RUSTFLAGS="-Ctarget-cpu=native" cargo build -q --release
```

3. Create conda env
```sh
conda env create --file scripts/hale-env.yml
conda activate hale
```


##  <a name="usage"></a>Usage

1. minimap2 alignment and batching
```shell
scripts/create_batched_alignments.sh <input_fastq/input_fastq.gz> <read_ids> <num_of_threads> <directory_for_batches_of_alignments> 
```
We use same parameters for minimap2 as HERRO <br>
Note: Read ids can be obtained with seqkit: ```seqkit seq -ni <input_fastq/input_fastq.gz> > <read_ids>```

3. Error-correction
```shell
hale correct --read-alns <directory_for_batches_of_alignments> -t 64 <input_fastq/input_fastq.gz> <fasta_output> 
```
Note: Flag ```-t``` represent number of threads.




##  <a name="ack"></a>Acknowledgements

This work leverages components of the HERRO framework, developed by Stanojevic et al. (2024) (bioRxiv, doi:10.1101/2024.05.18.594796). While we designed a new algorithm independent of HERRO's deep learning approach, we adopted key preprocessing steps such as Minimap2 alignment, windowing, and post-processing for consensus generation with minimal modifications. We are grateful to the HERRO authors for their valuable contribution to this field.







