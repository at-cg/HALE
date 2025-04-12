# HALE

HALE (Haplotype-Aware Long read Error correction)  is a haplotype-aware error-correction tool for PacBio Hifi reads and can be used for ONT Simplex reads as well. However the tools performance on ONT data still needs improvement to match its competitors like hale which uses Deep-learning approach t solve this problem.

## Requirements

- Linux OS (tested on RHEL 8.6 and Ubuntu 22.04)
- [Zstandard](https://facebook.github.io/zstd/)
- Python (and conda) for data preprocessing

### Compile from source

- [libtorch 2.0.*](https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu117.zip)
- [rustup](https://rustup.rs/)


## Installation

0. Clone the repository
```shell
git clone https://github.com/parveshbarak/HALE.git
cd HALE
```

1. Create conda environment
```shell
conda env create --file scripts/hale-env.yml
```

2. Build ```hale``` binary (ensure that libtorch and rustup are downloaded and installed.)

```shell
export LIBTORCH=<libtorch_path>
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
RUSTFLAGS="-Ctarget-cpu=native" cargo build -q --release
```
Path to the resulting binary: ```target/release/hale```

Compiling from source takes just a few minutes on a standard machine.

<!-- 
## Model Download

  1. Download model:
     
  For R10.4.1 data,
  ```shell
  wget -O model_R10_v0.1.pt https://zenodo.org/records/12683277/files/model_v0.1.pt?download=1
  ```
  For R9.4.1 data (experimental),
  ```shell
  wget -O model_R9_v0.1.pt https://zenodo.org/records/12683277/files/model_R9_v0.1.pt?download=1
  ```

Models can also be found on Zenodo: [https://zenodo.org/records/12683277](https://zenodo.org/records/12683277)

-->

## Usage

1. Preprocess reads
```shell
scripts/preprocess.sh <input_fastq> <output_prefix> <number_of_threads> <parts_to_split_job_into>
```
Note: Porechop loads all reads into memory, so the input may need to be split into multiple parts. Set <parts_to_split_job_into> to 1 if splitting is not needed. In Dorado v0.5, adapter trimming was added, so adapter trimming and splitting using Porechop and duplex tools will probably be removed in the future.

2. minimap2 alignment and batching

Although minimap2 can be run from the ```hale``` binary (omit --read-alns or use --write-alns to store batched alignments for future use).

```shell
scripts/create_batched_alignments.sh <output_from_reads_preprocessing> <read_ids> <num_of_threads> <directory_for_batches_of_alignments> 
```
Note: Read ids can be obtained with seqkit: ```seqkit seq -ni <reads> > <read_ids>```

3. Error-correction
```shell
hale inference --read-alns <directory_alignment_batches> -m "hale" <preprocessed_reads> <fasta_output> 
```
Note: The flag ```-m``` is for module which takes three valid entries namely "hale", "pih", "consensus". "pih" stands for passive informative sites handling The default option is "hale".


## Acknowledgements

This work leverages components of the HERRO framework, developed by Stanojevic et al. (2024) (bioRxiv, doi:10.1101/2024.05.18.594796). While we designed a new algorithm independent of HERRO's deep learning approach, we adopted key preprocessing steps such as Minimap2 alignment, windowing, and post-processing for consensus generation with minimal modifications. We are grateful to the HERRO authors for their valuable contribution to this field.


<!-- 
## Demo data

Instructions for downloading and running the demo data can be found [here](https://github.com/lbcb-sci/hale/blob/main/demo/README.txt).

## Citing hale

> Stanojevic, D., Lin, D., Florez De Sessions, P., & Sikic, M. (2024). Telomere-to-telomere phased genome assembly using error-corrected Simplex nanopore reads. bioRxiv, 2024-05. [doi:10.1101/2024.05.18.594796](https://doi.org/10.1101/2024.05.18.594796)

## Results on HG002 data

HG002 data was assembled using hifiasm and compared to HiFi reads. Results for uncorrected reads are not given since they produce poor assembly. Currently, data is not publicly available.

Assembly results and comparison with Hifi reads and uncorrected UL are given in the table below. Assemblies were perform using:

1. Hifi reads/Duplex ONT reads/Corrected UL reads
2. Uncorrected Ultra-long ONT reads as UL reads
3. Parental Illumina data

Hifiasm command used for all experiments:
```shell
hifiasm -o <output_prefix> -t <num_threads> --ul <UL_reads> --ul-cut 10000 -1 <parent1_yak> -2 <parent2_yak> <HiFi/Duplex/Corrected UL reads>
```

### Results

<img src='hg002-assm-results.png' alt="HG002 Assembly Results" width='600'>


## Results on Error-corrected HG002 experimental, high-accuracy, UL data

Experimental high-accuracy, UL HG002 error-corrected reads can be downloaded as below. 
Raw data used for the error-correction can be found [here](https://labs.epi2me.io/gm24385_ncm23_preview/). Assemblies were done in the same way as in the previous section.

### Download
  1. Download error-corrected reads:
  ```shell
  wget -O HG002.experimentalUL.corrected.fasta.gz https://zenodo.org/records/13252378/files/HG002.experimentalUL.corrected.fasta.gz?download=1
  ```

### Results

Assembly results and comparison with Hifi reads and uncorrected UL are given in the table below. Assemblies were perform using:

1. Hifi/HQ Uncorrected UL/Corrected UL reads
2. HQ Uncorrected Ultra-long ONT reads as UL reads
3. Parental Illumina data

<img src='hg002-hq-assm-results.png' alt="HG002 HQ Assembly Results" width='600'>


## Acknowledgements

This work has been supported by AI Singapore 100 Experiments (100E) Programme under the project AI-driven De Novo Diploid Assembler (AISG2-100E-2021-076) in collaboration with Agency for Science, Technology and Research (A*STAR), and Oxford Nanopore Technologies plc. (ONT). -->



<!-- TO do:

  1. Update the readme file.
  2. Make the code clean by removing unwated things [like generate features in main file]
  3. Make the code faster by better parallelism
  4. Update the code to take different running schemes as input like [consenus, original mec, hale] etc.
  5. 




 -->