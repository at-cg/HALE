use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use features::extract_features;
use haec_io::HAECRecord;

use pbars::{
    get_parse_reads_spinner, set_parse_reads_spinner_finish, track_progress, PBarNotification,
};

use glob::glob;
use rustc_hash::FxHashSet as HashSet;
use std::fs::metadata;
use std::io::Error;
use std::{
    fs::File,
    io,
    io::{prelude::*, BufWriter},
    path::Path,
    thread::{self},
};

use crate::{
    consensus::consensus_worker,
    features::CorrectOutput,
    correct::correct_worker,
    overlaps::alignment_reader,
};

mod aligners;
mod consensus;
mod features;
mod haec_io;
mod correct;
mod mm2;
mod overlaps;
mod pbars;
mod windowing;
mod hale_updated;

pub(crate) const READS_BATCH_SIZE: usize = 50_000;
pub(crate) const ALN_CHANNEL_CAPACITY: usize = 50_000;
pub(crate) const LINE_ENDING: u8 = b'\n';
pub(crate) const INFER_CHANNEL_CAP_FACTOR: usize = 2;

pub enum AlnMode<V: AsRef<Path>> {
    None,
    Read(V),
    Write(V),
}





pub fn error_correction<T, U, V>(
    reads_path: T,
    output_path: U,
    cluster_path: &str,
    window_size: u32,
    batch_size: usize,
    n_threads: usize,
    coverage: usize,
    ploidy: usize,
    aln_mode: AlnMode<V>,
    module: &str,
) where
    T: AsRef<Path> + Send + Sync,
    U: AsRef<Path> + Send + Sync,
    V: AsRef<Path> + Send,
{   
    let TOP_K = std::cmp::min(20,coverage/ploidy);


    let num_threads = n_threads;
    let mut n_correct = std::cmp::min(2, n_threads / 2);
    let mut n_extract = std::cmp::max(1, n_threads - n_correct);
    if(module=="hale") {
        n_correct = std::cmp::max(1, n_threads / 2);
        n_extract = n_threads.saturating_sub(n_correct);
    }
    // let num_threads = n_threads/2;
    // let n_correct = std::cmp::max(1, n_threads / 2);
    // let n_extract = n_threads.saturating_sub(n_correct);
    
    let num_consensus_threads = std::cmp::min(4,num_threads/16); 

    let (core, neighbour) = read_cluster(&cluster_path);
    let mut reads = parse_reads(&reads_path, window_size, &core, &neighbour);
    let max_len = reads.iter().map(|r| r.seq.len()).max().unwrap();

    let (alns_sender, alns_receiver) = bounded(ALN_CHANNEL_CAPACITY);
    let (writer_sender, writer_receiver) = unbounded();
    let (pbar_sender, pbar_receiver) = unbounded();

    thread::scope(|s| {
        let pbar_s = pbar_sender.clone();
        s.spawn(|| {
            alignment_reader(
                &reads,
                &reads_path,
                &core,
                aln_mode,
                num_threads,
                alns_sender,
                pbar_s,
            );
        });

        s.spawn(|| correction_writer_2(&reads, output_path, writer_receiver, pbar_sender));

        let (infer_sender, infer_recv) = bounded(INFER_CHANNEL_CAP_FACTOR * num_threads);
        let (cons_sender, cons_recv) = unbounded();

        for _ in 0..n_extract {
            let alns_r = alns_receiver.clone();
            let infer_s = infer_sender.clone();
            let ref_reads = &reads;

            s.spawn(move || {
                let mut feats_output = CorrectOutput::new(infer_s, batch_size);
                let mut tbuf = vec![0; max_len];
                let mut qbuf = vec![0; max_len];

                loop {
                    let (rid, alns) = match alns_r.recv() {
                        Ok(out) => out,
                        Err(_) => break,
                    };

                    extract_features(
                        rid,
                        ref_reads,
                        alns,
                        TOP_K,
                        module,
                        (&mut tbuf, &mut qbuf),
                        &mut feats_output,
                    );
                }
            });
        }
        
        
        for _ in 0..n_correct {
            let infer_recv_cloned = infer_recv.clone();
            let cons_sender_cloned = cons_sender.clone();
            
            s.spawn(move || {
                correct_worker(
                    module,
                    infer_recv_cloned,
                    cons_sender_cloned,
                )
            });
        }

        for _ in 0..num_consensus_threads {
            let c_recv = cons_recv.clone();
            let w_sender = writer_sender.clone();
            s.spawn(move || consensus_worker(c_recv, w_sender));
        }
        // --- CHANGE END ---

        drop(infer_sender);
        drop(cons_sender);
        
        drop(writer_sender); 

        track_progress(pbar_receiver);
    });
}


fn read_cluster(cluster_path: &&str) -> (Option<HashSet<String>>, Option<HashSet<String>>) {
    if !cluster_path.is_empty() {
        let file = match File::open(&cluster_path) {
            Ok(file) => file,
            Err(_) => panic!("Failed to open file: {:?}", cluster_path),
        };
        let reader = io::BufReader::new(file);
        let mut core: HashSet<String> = HashSet::default();
        let mut neighbour: HashSet<String> = HashSet::default();
        for line in reader.lines() {
            let line = match line {
                Ok(line) => line,
                Err(_) => panic!("Failed to read line: {:?}", line),
            };
            let fields: Vec<_> = line.split('\t').collect();
            match fields[0] {
                "0" => {
                    core.insert(fields[1].to_owned());
                }
                "1" => {
                    neighbour.insert(fields[1].to_owned());
                }
                _ => {
                    panic!("Invalid cluster file");
                }
            }
        }
        (Some(core), Some(neighbour))
    } else {
        (None, None)
    }
}

fn parse_reads<P: AsRef<Path>>(
    reads_path: P,
    window_size: u32,
    core: &Option<HashSet<String>>,
    neighbour: &Option<HashSet<String>>,
) -> Vec<HAECRecord> {
    // Get fastq reads
    let spinner = get_parse_reads_spinner(None);
    let md = metadata(&reads_path).unwrap();
    if md.is_file() {
        let reads = haec_io::get_reads(&reads_path, window_size, core, neighbour);
        set_parse_reads_spinner_finish(reads.len(), spinner);
        reads
    } else {
        let g = reads_path.as_ref().join("*").to_str().unwrap().to_owned();
        let reads: Vec<_> = glob(&g)
            .unwrap()
            .filter_map(|p| p.ok().and_then(|path| path.to_str().map(|s| s.to_owned())))
            .filter(|s| s.ends_with(".fastq") || s.ends_with(".fastq.gz"))
            .flat_map(|s| haec_io::get_reads(&s, window_size, core, neighbour))
            .collect();
        set_parse_reads_spinner_finish(reads.len(), spinner);
        reads
    }
}

fn correction_writer<U: AsRef<Path>>(
    reads: &[HAECRecord],
    output_path: U,
    consensus_recv: Receiver<(usize, Vec<Vec<u8>>)>,
    pbar_sender: Sender<PBarNotification>,
) {
    let file = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(file);

    loop {
        let (rid, seqs) = match consensus_recv.recv() {
            Ok(out) => out,
            Err(_) => break,
        };

        if seqs.len() == 1 {
            write_sequence(&seqs[0], None, &reads[rid], &mut writer).unwrap();
        } else {
            for (i, seq) in seqs.into_iter().enumerate() {
                write_sequence(&seq, Some(i), &reads[rid], &mut writer).unwrap();
            }
        }

        pbar_sender.send(PBarNotification::Inc).unwrap();
    }
}

fn write_sequence<W: Write>(
    seq: &[u8],
    idx: Option<usize>,
    read: &HAECRecord,
    writer: &mut W,
) -> Result<(), Error> {
    writer.write_all(b">")?;
    writer.write_all(&read.id)?;

    match idx {
        Some(idx) => write!(writer, ":{} ", idx)?,
        None => writer.write_all(b" ")?,
    };

    if let Some(desc) = read.description.as_ref() {
        writer.write_all(desc)?;
    }
    writer.write_all(b"\n")?;

    writer.write_all(&seq)?;
    write!(writer, "\n")?;

    Ok(())
}




fn split_seq_qual(buf: &[u8]) -> Result<(&[u8], &[u8]), Error> {
    let pos = buf.iter().position(|&b| b == b'+')
        .ok_or_else(|| Error::new(std::io::ErrorKind::InvalidData, "Missing '+' separator"))?;

    let (seq, qual) = buf.split_at(pos);
    let qual = &qual[1..]; // skip '+'

    if seq.len() != qual.len() {
        return Err(Error::new(
            std::io::ErrorKind::InvalidData,
            "Sequence and quality lengths differ",
        ));
    }

    Ok((seq, qual))
}


fn correction_writer_2<U: AsRef<Path>>(
    reads: &[HAECRecord],
    output_path: U,
    consensus_recv: Receiver<(usize, Vec<Vec<u8>>)>,
    pbar_sender: Sender<PBarNotification>,
) {
    let file = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(file);

    loop {
        let (rid, seqs) = match consensus_recv.recv() {
            Ok(out) => out,
            Err(_) => break,
        };
    
        if seqs.len() == 1 {
            let (seq, qual) = split_seq_qual(&seqs[0]).unwrap();
            write_sequence_fastq(seq, qual, None, &reads[rid], &mut writer).unwrap();
        } else {
            for (i, buf) in seqs.into_iter().enumerate() {
                let (seq, qual) = split_seq_qual(&buf).unwrap();
                write_sequence_fastq(seq, qual, Some(i), &reads[rid], &mut writer).unwrap();
            }
        }
    
        pbar_sender.send(PBarNotification::Inc).unwrap();
    }
}


fn write_sequence_fastq<W: Write>(
    seq: &[u8],
    qual: &[u8],
    idx: Option<usize>,
    read: &HAECRecord,
    writer: &mut W,
) -> Result<(), Error> {
    // Header
    writer.write_all(b"@")?;
    writer.write_all(&read.id)?;

    match idx {
        Some(idx) => write!(writer, ":{} ", idx)?,
        None => writer.write_all(b" ")?,
    };

    if let Some(desc) = read.description.as_ref() {
        writer.write_all(desc)?;
    }
    writer.write_all(b"\n")?;

    // Sequence
    writer.write_all(seq)?;
    writer.write_all(b"\n+\n")?;

    // Quality
    writer.write_all(qual)?;
    writer.write_all(b"\n")?;

    Ok(())
}

