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
  <!-- - To install these on Ubuntu/Debian: ``sudo apt-get install build-essential autoconf libtool pkg-config`` -->

  Note: If you're using a Linux system, there's a good chance these system packages are already installed, especially if development tools have been previously set up.



## <a name="started"></a>Try HALE on Small Test Data
The entire test workflow below will take about 5-6 minutes. Users can either run the commands one by one or copy the commands into an executable script.

```sh
# Install HALE 
git clone https://github.com/at-cg/HALE.git
cd HALE && RUSTFLAGS="-Ctarget-cpu=native" cargo build -q --release

# Create conda env
conda env create --file scripts/hale-env.yml
conda activate hale

mkdir -p test_run && cd test_run/

# download small test dataset
wget https://github.com/chhylp123/hifiasm/releases/download/v0.7/chr11-2M.fa.gz

# Run hale correction
../target/release/hale correct --reads chr11-2M.fa.gz --threads 8 --depth 40 --ploidy 2 --tech hifi

```

### <a name="visulize"></a>Visualize Corrections

**To visulize** the reduction in errors after HALE correction:
  - One can align both the raw and corrected reads to a high-quality reference genome (in this case, [HG002 v1.1](https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/HG002/assemblies/hg002v1.1.fasta.gz)).
  - Visulize the bam files in [IGV](https://igv.org).

**Exampple result:** For the above test dataset, we corrected using HALE and the following IGV snapshot captures a 125kb genomic region, shwoing the alignment of the raw and HALE-corrected reads to the HG002 assembly.
<p align="left"> <img src="scripts/igv_snapshot.png" alt="IGV comparison of raw vs corrected reads" width="800">


<em>Figure 1: IGV view showing the reduction in errros (colored bars) after HALE correction.</em> </p>



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
$hale_dir/target/release/hale correct --reads <input_fastq/fastq.gz/fasta/fasta.gz> --threads <num_of_threads> --depth <dataset_depth> --ploidy <genome_ploidy> --tech <hifi/ont>
```
**Output file :** ```hale_corrected_<input_fastq/input_fastq.gz/>``` will be created in the same directory.

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

