## <a name="intro"></a>Introduction

HALE (**H**aplotype-**A**ware **L**ong-read **E**rror correction) is a haplotype-aware error correction tool designed for long reads. It works with both ONT Simplex and PacBio HiFi data.


## <a name="requirements"></a> Requirements
- Linux OS (tested on RHEL 8 and Ubuntu 22.04)
- [rustup](https://rustup.rs/) (Install using [rustup](https://rustup.rs/), the recommended way to get Rust)
- Python 3.1 or above (and conda) for data preprocessing

  <!-- ### <a name="sys-requirements"></a> System Requirements -->
- Make sure the following system packages are installed (Linux):
  <!-- - build-essential (build-essential includes g++, make and standard headers like libc6-dev)
  - autoconf 
  - libtool 
  - pkg-config -->
  - build-essential, autoconf, libtool, pkg-config

  Note: If you're using a Linux system, there's a good chance these system packages are already installed, especially if development tools have been previously set up.



## <a name="started"></a>Try HALE on Small Test Data
The entire test workflow below will take about 6-8 minutes. Users can either run the commands one by one or copy the commands into an executable script.

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

# Run hale correction
../target/release/hale correct --reads HG002.chr19_10M_12M.fastq.gz --threads 16 --depth 40 --ploidy 2 --tech ont

```
For large inputs, users are recommended to increase the thread count depending on the number of the cores available for use. HALE takes about 44 minutes for all-vs-all overlap and 12 minutes for correction consuming ~90 GB RAM using 64 threads on a multicore [Perlmutter CPU-based node](https://docs.nersc.gov/systems/perlmutter/architecture/) to process 60x HiFi chr9 HG002 human genome dataset.




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


```shell
$hale_dir/target/release/hale correct --reads <input_fastq/input_fastq.gz> --threads <num_of_threads> --depth <dataset_depth> --ploidy <genome_ploidy> --tech <hifi/ont>
```
**Output file :** ```hale_corrected_<input_fastq/input_fastq.gz>``` will be created in the same directory.

Note: 
  - `$hale_dir` is the path to HALE directory
  - Flag ```--depth``` represent dataset dept, (default 60x), ```--ploidy``` represent ploidy of genome, (default 2)
  - Temporary alignment files are removed automatically upon successful completion.


## Implementation Notes:
#### Automatic pipeline selection

HALE automatically selects the appropriate correction pipeline based on the sequencing technology:

- **PacBio HiFi**
  - Single round of all-vs-all overlap
  - One HALE correction step

- **ONT Simplex**
  - Three rounds of all-vs-all overlap
  - Two pre-correction rounds (`pih` mode)
  - One final correction round (`hale` mode)

No additional flags are required beyond `--tech`.


#### Dependencies (handled via conda)

The conda environment installs:
- minimap2
- seqkit
- samtools
- Python dependencies for preprocessing


##  <a name="ack"></a>Acknowledgement

This work leverages components of [HERRO](https://github.com/lbcb-sci/herro), developed by Stanojevic et al. (2024). While we designed a new algorithm independent of HERRO's deep learning approach, we adopted key preprocessing steps such as Minimap2 alignment, and post-processing for consensus generation with a few modifications. 







