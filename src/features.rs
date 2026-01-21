use npyz::WriterBuilder;
use rustc_hash::FxHashMap as HashMap;
use std::fs::{create_dir_all, File};
use std::io::prelude::*;
use std::io::{BufWriter, Result};
use std::path::Path;
use std::cmp::min;

use crossbeam_channel::Sender;

use ndarray::{s, stack, Array, Array1, Array2, ArrayBase, ArrayViewMut1, Axis, Data, Ix2};
use ordered_float::OrderedFloat;

use crate::aligners::{CigarIter, CigarOp};
use crate::haec_io::HAECRecord;
use crate::correct::{prepare_examples, CorrectData, WindowExample};
use crate::overlaps::{Alignment, Strand};
use crate::pbars::PBarNotification;
use crate::windowing::{extract_windows, OverlapWindow};

// pub(crate) const TOP_K: usize = 20;
const MIN_COV_TH: u32 = 6;


const BASE_LOWER: [u8; 128] = [
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 97, 255, 99, 255, 255, 255, 103, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 116, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
];

const BASE_FORWARD: [u8; 128] = [
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 42, 255, 255,
    255, 255, 255, 255, 42, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 65, 255, 67, 255, 255, 255, 71, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 84, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 65, 255, 67, 255, 255, 255, 71, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 84, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
];

const ALIGNMENT_LEN_TH:usize = 4000;


// # for debug purpose:

#[derive(Debug)]
struct CoverageStats {
    min: usize,
    median: f64,
    mean: f64,
    mode: usize,
    p25: f64,
    p75: f64,
    p90: f64,
}

fn percentile(sorted: &[usize], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let rank = p * (n as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;

    if lo == hi {
        sorted[lo] as f64
    } else {
        let w = rank - lo as f64;
        sorted[lo] as f64 * (1.0 - w) + sorted[hi] as f64 * w
    }
}

fn coverage_stats(mut cov: Vec<usize>) -> CoverageStats {
    cov.sort_unstable();
    let n = cov.len();

    let min = cov[0];

    let mean = cov.iter().sum::<usize>() as f64 / n as f64;

    let median = if n % 2 == 0 {
        (cov[n / 2 - 1] + cov[n / 2]) as f64 / 2.0
    } else {
        cov[n / 2] as f64
    };

    let mut freq: HashMap<usize, usize> = HashMap::default();
    for &v in &cov {
        *freq.entry(v).or_insert(0) += 1;
    }
    
    let mode = freq
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(val, _)| val)
        .unwrap();

    CoverageStats {
        min,
        median,
        mean,
        mode,
        p25: percentile(&cov, 0.25),
        p75: percentile(&cov, 0.75),
        p90: percentile(&cov, 0.90),
    }
}


fn column_coverage(bases: &Array2<u8>) -> Vec<usize> {
    let (_rows, cols) = bases.dim();

    (0..cols)
        .map(|c| {
            bases.column(c)
                .iter()
                .filter(|&&b| b != b'.')
                .count()
        })
        .collect()
}




// Heuristics to filter rows from based on alignment accuracy
// Heuristic 3:
    // The core idea of Heuristic Three is to limit the column-wise coverage of non-gap bases, ensuring that at no position (col), 
    // the number of aligned (non-gap) segments passing through it exceeds a threshold (top_k = 20).
        // It processes rows sequentially and maintains a running coverage count for each column.
        // For each new row, it attempts to include the row, or portions of it, only if doing so doesn't push the column coverage above top_k.
        // If including a segment would violate the coverage constraint, that segment is "chopped" (excluded), and the function looks for subsequent, valid segments in the same row.
        // Valid segments are those that can be included without exceeding the top_k coverage limit and are at least ALIGNMENT_LEN_TH bases long.
        // The final output arrays, filtered_bases and filtered_quals, contain the selected and chopped segments, 
        // where excluded regions (or non-selected columns) are filled with the gap character (b'.').
// fn filter_rows_heuristic_three(bases: &Array2<u8>, quals: &Array2<u8>) -> (Array2<u8>, Array2<u8>) {
//     // let top_k = 20;
//     let ncols = bases.ncols();
//     let mut coverage = vec![0usize; ncols];

//     // let mut coverage_ = column_coverage(&bases);
//     // let stats = coverage_stats(coverage_);
//     // println!("{:#?}, dim: {:?}", stats, bases.dim());


//     let mut filtered_rows = Vec::new();
//     filtered_rows.push((0, 0, ncols-1));
//     // iterate over rows
//     for row in 1..bases.nrows() {
//         let mut is_chop = false;
//         // create a vector to store <start,end index pairs> in each row where coverage const are not violated
//         let mut valid_ranges = Vec::new();
//         let mut start_idx = 0;
//         let mut end_idx = 0;
//         let mut no_chop_start = 0;
//         let mut no_chop_end = ncols-1;
//         let mut first_non_gap_found = false;
//         {
//             let row_view = bases.row(row);
//             for (col, &val) in row_view.iter().enumerate() {
//                 if val != b'.' {
//                     if !first_non_gap_found {
//                         first_non_gap_found = true;
//                         no_chop_start = col;
//                         start_idx = col;
//                         end_idx = col;
//                     }
//                     if coverage[col] + 1 > TOP_K {
//                         is_chop = true;
//                         if end_idx - start_idx >= ALIGNMENT_LEN_TH {
//                             valid_ranges.push((start_idx, end_idx-1));
//                         }
//                         start_idx = col + 1;
//                     }
//                     end_idx += 1;
//                 } else {
//                     if first_non_gap_found {
//                         no_chop_end = col - 1;
//                         break;
//                     }
//                 }
//             }
//             if !is_chop {
//                 filtered_rows.push((row, no_chop_start, no_chop_end)); // keep the whole row
//                 // add 1 to all indices in range
//                 for i in no_chop_start..=no_chop_end {
//                     coverage[i] += 1;
//                 }
//             } else if is_chop && valid_ranges.len() > 0 {
//                 for (start, end) in valid_ranges {
//                     filtered_rows.push((row, start, end));
//                     for i in start..=end {
//                         coverage[i] += 1;
//                     }
//                 }
//             }
//         }
//     }

//     let mut filtered_bases = Array2::zeros((filtered_rows.len(), ncols));
//     let mut filtered_quals = Array2::zeros((filtered_rows.len(), ncols));
//     let mut row_idx = 0;
//     for &(row, start, end) in &filtered_rows {
//         for col in 0..ncols {
//             if col >= start && col <= end {
//                 filtered_bases[(row_idx, col)] = bases[(row, col)];
//                 filtered_quals[(row_idx, col)] = quals[(row, col)];
//             } else {
//                 filtered_bases[(row_idx, col)] = b'.';
//             }
//         }
//         row_idx += 1;
//     }

//     (filtered_bases, filtered_quals)
// }







fn get_max_ins_for_window(
    overlaps: &[OverlapWindow], // Sorted overlaps
    ovlps_cigar_map: &HashMap<u32, &Vec<u8>>,
    tid: u32,
    window_length: usize,
) -> Vec<u16> {
    let mut max_ins = vec![0; window_length];
    for ow in overlaps.iter() {
        let is_target = ow.overlap.tid == tid;
        if is_target {
            let mut tpos = ow.overlap.tstart as usize;
            // Handle cigar
            let qid = ow.overlap.return_other_id(tid);
            let cigar = ovlps_cigar_map.get(&qid).unwrap();
            //let cigar_len = ow.cigar_end_idx - ow.cigar_start_idx + 1;
            let cigar_iter = CigarIter::new(&cigar);

            cigar_iter.for_each(|(op, range)| {
                let l = match op {
                    CigarOp::Match(l) | CigarOp::Mismatch(l) | CigarOp::Deletion(l) => l as usize,
                    CigarOp::Insertion(l) => {
                        assert!(
                            tpos <= max_ins.len(),
                            "Length {} is bigger than the tseq {}. {} {} {} {:?} {:?}",
                            tpos,
                            max_ins.len(),
                            ow.cigar_start_offset,
                            op.get_length(),
                            std::str::from_utf8(cigar).unwrap(),
                            range,
                            ow.cigar_start_idx..ow.cigar_end_idx
                        );

                        max_ins[tpos - 1] = max_ins[tpos - 1].max(l as u16);
                        return;
                    }
                };
                tpos += l;
            });
        } else {
            let mut tpos = ow.overlap.qstart as usize;
            // Handle cigar
            let qid = ow.overlap.return_other_id(tid);
            let cigar = ovlps_cigar_map.get(&qid).unwrap();
            //let cigar_len = ow.cigar_end_idx - ow.cigar_start_idx + 1;
            let cigar_iter = CigarIter::new(&cigar);

            cigar_iter.for_each(|(op, range)| {
                let l = match op {
                    CigarOp::Match(l) | CigarOp::Mismatch(l) | CigarOp::Insertion(l) => l as usize,
                    CigarOp::Deletion(l) => {
                        assert!(
                            tpos <= max_ins.len(),
                            "Length {} is bigger than the tseq {}. {} {} {} {:?} {:?}",
                            tpos,
                            max_ins.len(),
                            ow.cigar_start_offset,
                            op.get_length(),
                            std::str::from_utf8(cigar).unwrap(),
                            range,
                            ow.cigar_start_idx..ow.cigar_end_idx
                        );

                        max_ins[tpos - 1] = max_ins[tpos - 1].max(l as u16);
                        return;
                    }
                };
                tpos += l;
            });
        }
    }

    max_ins
}




fn get_query_region(window: &OverlapWindow, tid: u32) -> (u32, u32) {
    let (qstart, qend) = if window.overlap.tid == tid {
        (window.overlap.qstart, window.overlap.qend)
    } else {
        (window.overlap.tstart, window.overlap.tend)
    };

    match window.overlap.strand {
        Strand::Forward => (qstart + window.qstart, qstart + window.qend),
        Strand::Reverse => (qend - window.qend, qend - window.qstart),
    }
}

fn get_features_for_ol_window(
    mut bases: ArrayViewMut1<'_, u8>,
    mut quals: ArrayViewMut1<'_, u8>,
    window: &OverlapWindow,
    cigar: &[u8],
    query: &HAECRecord,
    tid: u32,
    max_ins: &[u16],
    qbuffer: &mut [u8],
) {
    // Handle query sequence
    let (qstart, qend) = if window.overlap.tid == tid {
        (window.overlap.qstart, window.overlap.qend)
    } else {
        (window.overlap.tstart, window.overlap.tend)
    };

    let (tstart, tend) = if window.overlap.tid != tid {
        (window.overlap.qstart, window.overlap.qend)
    } else {
        (window.overlap.tstart, window.overlap.tend)
    };

    let is_target = window.overlap.tid == tid;

    let mut query_iter: Box<dyn DoubleEndedIterator<Item = (&u8, &u8)>> =
        match window.overlap.strand {
            Strand::Forward => {
                let range = qstart as usize..qend as usize;
                let qlen = qend as usize - qstart as usize;

                query.seq.get_subseq(range.clone(), qbuffer);
                let quals = &query.qual[range];

                Box::new(qbuffer[..qlen].iter().zip(quals))
            }
            Strand::Reverse => {
                let range = qstart as usize..qend as usize;
                let qlen = qend as usize - qstart as usize;

                query.seq.get_rc_subseq(range.clone(), qbuffer);
                let quals = &query.qual[range];

                Box::new(
                    qbuffer[..qlen]
                        .iter()
                        .zip(quals.iter().rev())
                        .map(|(b, q)| (&BASE_LOWER[*b as usize], q)),
                )
            }
        };
    //let mut query_iter = query_iter.skip(window.qstart as usize);

    // Number of cigars for the window
    // TODO get error when we calculate correct number for end -> (idx, 0)
    // Works for this expression but unecessarily iterates through (idx, 0)
    //let cigar_len = window.cigar_end_idx - window.cigar_start_idx + 1;
    //let cigar_end = cigar.len().min((window.cigar_end_idx + 1) as usize);

    // Handle cigar
    //let cigar = cigar[window.cigar_start_idx as usize..cigar_end].iter();
    let cigar = CigarIter::new(&cigar);

    // Get features
    let gap = if let Strand::Forward = window.overlap.strand {
        b'*'
    } else {
        b'#'
    };
    bases.fill(gap); // Initialize with gap token

    let mut tpos = tstart as usize; // position in the target read (excluding insertions)
    let mut idx = tstart as usize + max_ins[..tstart as usize].iter().map(|v| *v as usize).sum::<usize>(); // position in the features (including insertions)

    if idx > 0 {
        // No alignment at the start
        bases.slice_mut(s![..idx]).fill(b'.');
    }

    cigar.for_each(|(op, range)| {
        let mut l = match op {
            CigarOp::Match(l)
            | CigarOp::Mismatch(l)
            | CigarOp::Deletion(l)
            | CigarOp::Insertion(l) => l as usize,
        };

        // Write features
        if is_target {
            match op {
                CigarOp::Match(_) | CigarOp::Mismatch(_) => {
                    for i in 0..l {
                        let (base, qual) = query_iter
                            .next()
                            .expect("Base and its quality should be present.");
                        bases[idx] = *base;
                        quals[idx] = *qual;

                        idx += 1 + max_ins[tpos + i] as usize;
                    }

                    tpos += l;
                }
                CigarOp::Deletion(_) => {
                    for i in 0..l {
                        // No need to write gap, gap is already written
                        idx += 1 + max_ins[tpos + i] as usize;
                    }

                    tpos += l;
                }
                CigarOp::Insertion(_) => {
                    /*assert!(
                        max_ins[tpos - 1] as usize >= l,
                        "Insertion length is bigger than max_ins"
                    );*/

                    idx -= max_ins[tpos - 1] as usize; // Return to first insertion for the previous base
                    for i in 0..l {
                        let (base, qual) = query_iter
                            .next()
                            .expect("Base and its quality should be present.");

                        bases[idx + i] = *base;
                        quals[idx + i] = *qual;
                    }
                    idx += max_ins[tpos - 1] as usize; // Move back to the last base
                }
            }
        } else {
            match op {
                CigarOp::Match(_) | CigarOp::Mismatch(_) => {
                    for i in 0..l {
                        let (base, qual) = query_iter
                            .next()
                            .expect("Base and its quality should be present.");
                        bases[idx] = *base;
                        quals[idx] = *qual;

                        idx += 1 + max_ins[tpos + i] as usize;
                    }

                    tpos += l;
                }
                CigarOp::Insertion(_) => {
                    for i in 0..l {
                        // No need to write gap, gap is already written
                        idx += 1 + max_ins[tpos + i] as usize;
                    }

                    tpos += l;
                }
                CigarOp::Deletion(_) => {
                    /*assert!(
                        max_ins[tpos - 1] as usize >= l,
                        "Insertion length is bigger than max_ins"
                    );*/

                    idx -= max_ins[tpos - 1] as usize; // Return to first insertion for the previous base
                    for i in 0..l {
                        let (base, qual) = query_iter
                            .next()
                            .expect("Base and its quality should be present.");

                        bases[idx + i] = *base;
                        quals[idx + i] = *qual;
                    }
                    idx += max_ins[tpos - 1] as usize; // Move back to the last base
                }
            }
        }
    });

    if idx < bases.shape()[0] {
        // No alignment at the end
        bases.slice_mut(s![idx..]).fill(b'.');
    }
}

fn write_target_for_window(
    target: &HAECRecord,
    max_ins: &[u16],
    mut bases: ArrayViewMut1<'_, u8>,
    mut quals: ArrayViewMut1<'_, u8>,
    window_length: usize,
    tbuffer: &[u8],
) {
    bases.fill(b'*'); // Fill like forward

    let mut tpos = 0;
    tbuffer
        .iter()
        .zip(target.qual.iter())
        .enumerate()
        .for_each(|(i, (b, q))| {
            bases[tpos] = *b;
            quals[tpos] = *q;

            tpos += 1 + max_ins[i] as usize;
        });
}

fn get_features_for_window(
    overlaps: &mut [OverlapWindow],
    ovlps_cigar_map: &HashMap<u32, &Vec<u8>>,
    tid: u32,
    reads: &[HAECRecord],
    max_ins: &[u16],
    window_length: usize, // Full window length
    tbuffer: &[u8],
    qbuffer: &mut [u8],
) -> (Array2<u8>, Array2<u8>) {
    //Get features
    let length = max_ins.iter().map(|v| *v as usize).sum::<usize>() + max_ins.len();
    let width = 1 + overlaps.len();
    // let width = 1 + min(20, overlaps.len());

    let mut bases = Array::from_elem((length, width), b'.');
    let mut quals = Array::from_elem((length, width), b'!');

    // First write the target
    write_target_for_window(
        &reads[tid as usize],
        &max_ins,
        bases.index_axis_mut(Axis(1), 0),
        quals.index_axis_mut(Axis(1), 0),
        window_length,
        tbuffer,
    );

    // Write top-k overlaps for the window
    // overlaps.iter().take(width-1).enumerate().for_each(|(i, ow)| {
    overlaps.iter().enumerate().for_each(|(i, ow)| {
        let qid = ow.overlap.return_other_id(tid);
        get_features_for_ol_window(
            bases.index_axis_mut(Axis(1), i + 1),
            quals.index_axis_mut(Axis(1), i + 1),
            ow,
            ovlps_cigar_map.get(&qid).unwrap(),
            &reads[qid as usize],
            tid,
            &max_ins,
            qbuffer,
        )
    });


    (bases, quals)
}










// use rustc_hash::FxHashMap as HashMap;
// use ndarray::{s, Array, Array1, Array2, ArrayViewMut1, Axis};
// // Assuming HAECRecord, OverlapWindow, ALIGNMENT_LEN_TH, etc. are available in scope

// fn get_features_for_window_filtered_prev(
//     overlaps: &mut [OverlapWindow],
//     ovlps_cigar_map: &HashMap<u32, &Vec<u8>>,
//     tid: u32,
//     reads: &[HAECRecord],
//     max_ins: &[u16],
//     window_length: usize, 
//     tbuffer: &[u8],
//     qbuffer: &mut [u8],
// ) -> (Array2<u8>, Array2<u8>) {
    
//     // 1. Calculate dimensions
//     let length = max_ins.iter().map(|v| *v as usize).sum::<usize>() + max_ins.len();
    
//     // 2. Setup reusable temporary buffers
//     let mut temp_bases = Array1::from_elem(length, b'.');
//     let mut temp_quals = Array1::from_elem(length, b'!');
    
//     // 3. Setup coverage tracker
//     let mut coverage = vec![0usize; length];
    
//     // 4. Output storage
//     let mut accepted_bases_flat = Vec::new();
//     let mut accepted_quals_flat = Vec::new();
//     let mut accepted_count = 0;

//     let mut push_segment = |bases_col: &Array1<u8>, quals_col: &Array1<u8>, start: usize, end: usize| {
//         for i in 0..length {
//             if i >= start && i <= end {
//                 accepted_bases_flat.push(bases_col[i]);
//                 accepted_quals_flat.push(quals_col[i]);
//             } else {
//                 accepted_bases_flat.push(b'.');
//                 accepted_quals_flat.push(b'!');
//             }
//         }
//         accepted_count += 1;
//     };

//     // --- Step A: Process Target ---
//     write_target_for_window(
//         &reads[tid as usize],
//         &max_ins,
//         temp_bases.view_mut(),
//         temp_quals.view_mut(),
//         window_length,
//         tbuffer,
//     );

//     push_segment(&temp_bases, &temp_quals, 0, length - 1);
//     for c in &mut coverage { *c += 1; }

//     let mut cov_not_updated_ct = 0;

//     // --- Step B: Process Overlaps with Early Exit ---
//     for (_i, ow) in overlaps.iter().enumerate() {
//         temp_bases.fill(b'.');
//         temp_quals.fill(b'!');

//         let qid = ow.overlap.return_other_id(tid);
        
//         get_features_for_ol_window(
//             temp_bases.view_mut(),
//             temp_quals.view_mut(),
//             ow,
//             ovlps_cigar_map.get(&qid).unwrap(),
//             &reads[qid as usize],
//             tid,
//             &max_ins,
//             qbuffer,
//         );

//         let mut is_chop = false;
//         let mut valid_ranges = Vec::new();
//         let mut start_idx = 0;
//         let mut end_idx = 0;
//         let mut no_chop_start = 0; 
//         let mut no_chop_end = length - 1; 
//         let mut first_non_gap_found = false;

//         for (col, &val) in temp_bases.iter().enumerate() {
//             if val != b'.' {
//                 if !first_non_gap_found {
//                     first_non_gap_found = true;
//                     no_chop_start = col;
//                     start_idx = col;
//                     end_idx = col;
//                 }
                
//                 if coverage[col] + 1 > TOP_K {
//                     is_chop = true;
//                     // Use Constant for heuristic check
//                     if end_idx - start_idx >= ALIGNMENT_LEN_TH {
//                         valid_ranges.push((start_idx, end_idx - 1));
//                     }
//                     start_idx = col + 1;
//                 }
//                 end_idx += 1; 
//             } else {
//                 if first_non_gap_found {
//                     no_chop_end = col - 1;
//                     break; 
//                 }
//             }
//         }

//         let mut coverage_updated = false;

//         if !is_chop {
//             if first_non_gap_found {
//                 push_segment(&temp_bases, &temp_quals, no_chop_start, no_chop_end);
//                 for k in no_chop_start..=no_chop_end { coverage[k] += 1; }
//                 coverage_updated = true;
//             }
//         } else if is_chop && !valid_ranges.is_empty() {
//             for (start, end) in valid_ranges {
//                 push_segment(&temp_bases, &temp_quals, start, end);
//                 for k in start..=end { coverage[k] += 1; }
//             }
//             coverage_updated = true;
//         }

//         // -------------------------------------------------------------
//         // EARLY EXIT CHECK
//         // -------------------------------------------------------------
//         if coverage_updated {
//             let mut max_low_cov_run = 0;
//             let mut current_run = 0;

//             for &c in &coverage {
//                 if c < TOP_K {
//                     current_run += 1;
//                 } else {
//                     if current_run > max_low_cov_run {
//                         max_low_cov_run = current_run;
//                     }
//                     current_run = 0;
//                 }
//             }
//             // Check trailing run
//             if current_run > max_low_cov_run {
//                 max_low_cov_run = current_run;
//             }

//             // If the largest coverage gap is smaller than the threshold, stop.
//             if max_low_cov_run < ALIGNMENT_LEN_TH {
//                 // println!("hii: {:?}", _i);
//                 break;
//             }
//         } else {
//             cov_not_updated_ct += 1;
//         }
        
//         // if(cov_not_updated_ct > 10) {
//         //     break;
//         // }
//     }

//     // --- Step C: Reshape and Return ---
//     let bases_t = Array2::from_shape_vec((accepted_count, length), accepted_bases_flat)
//         .expect("Shape mismatch in bases generation");
//     let quals_t = Array2::from_shape_vec((accepted_count, length), accepted_quals_flat)
//         .expect("Shape mismatch in quals generation");

//     (bases_t.t().to_owned(), quals_t.t().to_owned())
// }







fn get_features_for_window_filtered(
    overlaps: &mut [OverlapWindow],
    ovlps_cigar_map: &HashMap<u32, &Vec<u8>>,
    tid: u32,
    reads: &[HAECRecord],
    max_ins: &[u16],
    window_length: usize,
    tbuffer: &[u8],
    qbuffer: &mut [u8],
    TOP_K: usize,     // Pass TOP_K as arg
    // aln_len_th: usize // Pass ALIGNMENT_LEN_TH as arg
) -> (Array2<u8>, Array2<u8>) {
    
    // 1. Calculate dimensions and Pre-calculate Insertion Prefix Sums
    //    This allows us to map tpos -> feature_idx in O(1)
    let length = max_ins.iter().map(|v| *v as usize).sum::<usize>() + max_ins.len();
    
    let mut ins_prefix_sum = vec![0usize; max_ins.len() + 1];
    let mut current_sum = 0;
    for (i, &val) in max_ins.iter().enumerate() {
        current_sum += val as usize;
        ins_prefix_sum[i + 1] = current_sum;
    }

    // 2. Setup reusable temporary buffers
    let mut temp_bases = Array1::from_elem(length, b'.');
    let mut temp_quals = Array1::from_elem(length, b'!');

    // 3. Setup coverage tracker
    let mut coverage = vec![0usize; length];

    // 4. Output storage
    let mut accepted_bases_flat = Vec::new();
    let mut accepted_quals_flat = Vec::new();
    let mut accepted_count = 0;

    // Helper to push segments
    let mut push_segment = |bases_col: &Array1<u8>, quals_col: &Array1<u8>, start: usize, end: usize| {
        for i in 0..length {
            if i >= start && i <= end {
                accepted_bases_flat.push(bases_col[i]);
                accepted_quals_flat.push(quals_col[i]);
            } else {
                accepted_bases_flat.push(b'.');
                accepted_quals_flat.push(b'!');
            }
        }
        accepted_count += 1;
    };

    // --- Step A: Process Target ---
    write_target_for_window(
        &reads[tid as usize],
        &max_ins,
        temp_bases.view_mut(),
        temp_quals.view_mut(),
        window_length,
        tbuffer,
    );

    push_segment(&temp_bases, &temp_quals, 0, length - 1);
    for c in &mut coverage { *c += 1; }

    let mut cov_not_updated_ct = 0;

    // --- Step B: Process Overlaps ---
    for (_i, ow) in overlaps.iter().enumerate() {
        // [OPTIMIZATION START] -----------------------------------------------
        // Before generating features, check if this read covers ANY useful region.
        
        // 1. Determine Target Range (tstart, tend)
        // Logic copied from get_features_for_ol_window to match coordinate system
        let (tstart, tend) = if ow.overlap.tid != tid {
            (ow.overlap.qstart, ow.overlap.qend)
        } else {
            (ow.overlap.tstart, ow.overlap.tend)
        };

        // 2. Map Target Range to Feature Index Range
        // Feature Idx = tpos + Sum(max_ins before tpos)
        // We use min(length) to ensure we don't go out of bounds if offsets are weird
        let idx_start = (tstart as usize + ins_prefix_sum[tstart as usize]).min(length);
        let idx_end = (tend as usize + ins_prefix_sum[tend as usize]).min(length);

        // 3. Scan Coverage in this projected range
        // We look for a contiguous gap of size >= ALIGNMENT_LEN_TH where coverage < top_k
        let mut potential_useful_len = 0;
        let mut is_worth_processing = false;

        for k in idx_start..idx_end {
            if coverage[k] < TOP_K {
                potential_useful_len += 1;
                // If we find ONE valid segment long enough, the read is worth parsing
                if potential_useful_len >= ALIGNMENT_LEN_TH {
                    is_worth_processing = true;
                    break;
                }
            } else {
                potential_useful_len = 0;
            }
        }

        // 4. Skip if useless
        if !is_worth_processing {
            // cov_not_updated_ct += 1;
            // // Early exit check inside the skip block
            // // (If we skip many times, we might want to check if the window is full)
            //  if cov_not_updated_ct > 500 { // Check occasionally
            //     let mut max_low_cov_run = 0;
            //     let mut current_run = 0;
            //     for &c in &coverage {
            //         if c < top_k { current_run += 1; } 
            //         else {
            //             if current_run > max_low_cov_run { max_low_cov_run = current_run; }
            //             current_run = 0;
            //         }
            //     }
            //     if current_run > max_low_cov_run { max_low_cov_run = current_run; }
                
            //     if max_low_cov_run < ALIGNMENT_LEN_TH { break; }
            //     cov_not_updated_ct = 0; // Reset counter
            // }
            continue;
        }
        // [OPTIMIZATION END] -------------------------------------------------

        // Reset buffers
        temp_bases.fill(b'.');
        temp_quals.fill(b'!');

        let qid = ow.overlap.return_other_id(tid);
        
        get_features_for_ol_window(
            temp_bases.view_mut(),
            temp_quals.view_mut(),
            ow,
            ovlps_cigar_map.get(&qid).unwrap(),
            &reads[qid as usize],
            tid,
            &max_ins,
            qbuffer,
        );

        // --- Heuristic Three Filter Logic ---
        let mut is_chop = false;
        let mut valid_ranges = Vec::new();
        let mut start_idx = 0;
        let mut end_idx = 0;
        let mut no_chop_start = 0; 
        let mut no_chop_end = length - 1; 
        let mut first_non_gap_found = false;

        for (col, &val) in temp_bases.iter().enumerate() {
            if val != b'.' {
                if !first_non_gap_found {
                    first_non_gap_found = true;
                    no_chop_start = col;
                    start_idx = col;
                    end_idx = col;
                }
                
                if coverage[col] + 1 > TOP_K {
                    is_chop = true;
                    if end_idx - start_idx >= ALIGNMENT_LEN_TH {
                        valid_ranges.push((start_idx, end_idx - 1));
                    }
                    start_idx = col + 1;
                }
                end_idx += 1; 
            } else {
                if first_non_gap_found {
                    no_chop_end = col - 1;
                    break; 
                }
            }
        }

        let mut coverage_updated = false;

        if !is_chop {
            if first_non_gap_found {
                push_segment(&temp_bases, &temp_quals, no_chop_start, no_chop_end);
                for k in no_chop_start..=no_chop_end { coverage[k] += 1; }
                coverage_updated = true;
            }
        } else if is_chop && !valid_ranges.is_empty() {
            for (start, end) in valid_ranges {
                push_segment(&temp_bases, &temp_quals, start, end);
                for k in start..=end { coverage[k] += 1; }
            }
            coverage_updated = true;
        }

        if coverage_updated {
            cov_not_updated_ct = 0; // Reset consecutive skip counter
            
            // Global Saturation Check (Existing logic)
            let mut max_low_cov_run = 0;
            let mut current_run = 0;

            for &c in &coverage {
                if c < TOP_K {
                    current_run += 1;
                } else {
                    if current_run > max_low_cov_run {
                        max_low_cov_run = current_run;
                    }
                    current_run = 0;
                }
            }
            if current_run > max_low_cov_run { max_low_cov_run = current_run; }

            if max_low_cov_run < ALIGNMENT_LEN_TH {
                break;
            }
        } else {
            cov_not_updated_ct += 1;
        }
    }

    let bases_t = Array2::from_shape_vec((accepted_count, length), accepted_bases_flat)
        .expect("Shape mismatch in bases generation");
    let quals_t = Array2::from_shape_vec((accepted_count, length), accepted_quals_flat)
        .expect("Shape mismatch in quals generation");

    (bases_t.t().to_owned(), quals_t.t().to_owned())
}











fn overlap_window_filter(cigar: &[u8]) -> bool {
    let long_indel = CigarIter::new(cigar).any(|(op, _)| match op {
        CigarOp::Insertion(l) | CigarOp::Deletion(l) if l >= 30 => true,
        _ => false,
    });

    !long_indel
}

pub(crate) fn extract_features<'a, T: FeaturesOutput<'a>>(
    rid: u32,
    reads: &'a [HAECRecord],
    overlaps: Vec<Alignment>,
    TOP_K: usize,
    module: &str,
    (tbuf, qbuf): (&mut [u8], &mut [u8]),
    feats_output: &mut T,
) {
    let read = &reads[rid as usize];
    reads[rid as usize].seq.get_sequence(tbuf);

    let window_size = read.seq.len();

    // Get overlaps for windows
    let n_windows = (read.seq.len() + window_size as usize - 1) / window_size as usize;
    let mut windows = vec![Vec::new(); n_windows];

    let mut ovlps_cigar_map = HashMap::default();
    for alignment in overlaps.iter() {
        let qid = alignment.overlap.return_other_id(rid);

        // let (tshift, qshift) = (0, 0);

        //Extract windows
        let is_target = alignment.overlap.tid == rid;
        extract_windows(
            &mut windows,
            &alignment.overlap,
            &alignment.cigar,
            // tshift,
            // qshift,
            is_target,
            // window_size,
        );

        ovlps_cigar_map.insert(qid, &alignment.cigar);

    }

    feats_output.init(rid, &read.id);
    for i in 0..n_windows {

        let win_len = read.seq.len();

        // Sort window to take TOP-K
        // Since there is just one window and for that window all information is in overlap, let not consider window specific variables.
        windows[i].sort_by_key(|ow| {
            let cigar = ovlps_cigar_map
                .get(&ow.overlap.return_other_id(rid))
                .unwrap();

            let tstart = ow.overlap.tstart as usize;
            let tend = ow.overlap.tend as usize;
            //reads[rid as usize].seq.get_subseq(tstart..tend, tbuf);

            let qid = ow.overlap.return_other_id(rid);
            let is_target = ow.overlap.tid == rid;
            // let (qstart, qend) = get_query_region(ow, rid);
            let qstart = ow.overlap.qstart as usize;
            let qend = ow.overlap.qend as usize;
            let qlen = (qend - qstart) as usize;
            match ow.overlap.strand {
                Strand::Forward => reads[qid as usize]
                    .seq
                    .get_subseq(qstart as usize..qend as usize, qbuf),
                Strand::Reverse => reads[qid as usize]
                    .seq
                    .get_rc_subseq(qstart as usize..qend as usize, qbuf),
            }
            let acc = 
                if is_target {
                    calculate_accuracy(ow, cigar, &tbuf[tstart..tend], &qbuf[..qlen])
                } else {
                    calculate_accuracy(ow, cigar, &qbuf[..qlen], &tbuf[tstart..tend])
                };
            OrderedFloat(-acc)
        });


        let max_ins = get_max_ins_for_window(
            &windows[i],
            &ovlps_cigar_map,
            rid,
            win_len,
        );

        let (full_bases, full_quals) = get_features_for_window_filtered(
            &mut windows[i],
            &ovlps_cigar_map,
            rid,
            reads,
            &max_ins,
            win_len,
            tbuf,
            qbuf,
            TOP_K,
        );

        // let full_bases_t = full_bases.t().to_owned();
        // let full_quals_t = full_quals.t().to_owned();
        // let (bases_t, quals_t) = filter_rows_heuristic_three(&full_bases_t, &full_quals_t);
        // let bases = bases_t.t().to_owned();
        // let quals = quals_t.t().to_owned();

        let bases = full_bases.to_owned();
        let quals = full_quals.to_owned();

        // println!("full bases dim: {:#?} \n bases dim {:?}", full_bases_t, bases_t.dim());
        
        // println!("All overlapping read ids:\n {:?}", rid);
        // for ow in &windows[i] {
        //     println!("{:?}", ow.overlap.return_other_id(rid));
        // }

        
        // println!("Read ids in top 20:\n {:?}", rid);
        // let mut iter = 0;
        // for j in 0..windows[i].len() {
        //     while (iter < selected_rows.len() && selected_rows[iter].0 == j) {
        //         let ow = windows[i][j].clone();
        //         println!("{:?}", ow.overlap.return_other_id(rid));
        //         iter += 1;
        //     }
        // }

        // println!("Read ids in top 20:\n {:?}", rid);
        // let mut iter = 0;
        // for j in 0..windows[i].len() {
        //     while (iter < 6 && selected_rows[iter].0 == j) {
        //         let ow = windows[i][j].clone();
        //         println!("{:?}", ow.overlap.return_other_id(rid));
        //         iter += 1;
        //     }
        // }



        // let mut out = String::new();
        // out.push_str(&format!("All overlapping read ids for {:?}: ", rid));
        // for ow in &windows[i] {
        //     out.push_str(&format!("{:?},", ow.overlap.return_other_id(rid)));
        // }
        // print!("{out}");


        // let mut out = String::new();
        // out.push_str(&format!("Read ids in top 20 for {:?}: ", rid));
        // let mut iter = 1;
        // for j in 0..windows[i].len() {
        //     while (iter < selected_rows.len() && selected_rows[iter].0 == j+1) {
        //         let ow = windows[i][j].clone();
        //         out.push_str(&format!("{:?},", ow.overlap.return_other_id(rid)));
        //         iter += 1;
        //     }
        // }
        // print!("{out}");

        // assert!(
        //     selected_rows.len() == bases.ncols(),
        //     "selected_rows len {} does not match bases rows {}",
        //     selected_rows.len(),
        //     bases.ncols()
        // );



        let qids: Vec<&str> = windows[i]
            .iter()
            .map(|ow| {
                std::str::from_utf8(&reads[ow.overlap.return_other_id(rid) as usize].id).unwrap()
            })
            .collect();

        // let (supported, weakly_supported) = get_supported(&bases, &quals, module);
        let supported = get_supported(&bases, module);



        let qids_test: Vec<u32> = windows[i].iter().map(|ow| ow.overlap.qid).collect();

        feats_output.update(
            rid,
            i as u16,
            bases,
            quals,
            qids_test,
            supported,
            qids,
            n_windows as u16,
        );
    }

    feats_output.emit();
}



fn calculate_accuracy(window: &OverlapWindow, cigar: &[u8], tseq: &[u8], qseq: &[u8]) -> f32 {
    let (mut tpos, mut qpos) = (0, 0);
    let (mut m, mut s, mut i, mut d) = (0, 0, 0, 0);

    let cigar_iter = CigarIter::new(&cigar);
    for (op, range) in cigar_iter {
        let len = op.get_length() as usize;

        assert!(len > 0, "Operation length cannot be 0");

        // Not insertion -> consume tseq -> check bounds
        if !matches!(op, CigarOp::Insertion(_)) {
            assert!(
                tpos + len <= tseq.len(),
                "Length {} + {} is bigger than the tseq {}. {} {} {} {:?} {:?}",
                len,
                tpos,
                tseq.len(),
                window.cigar_start_offset,
                op.get_length(),
                std::str::from_utf8(cigar).unwrap(),
                range,
                window.cigar_start_idx..window.cigar_end_idx
            );
        }

        // Not deletion -> consume qseq -> check bounds
        if !matches!(op, CigarOp::Deletion(_)) {
            assert!(
                qpos + len <= qseq.len(),
                "Length {} + {} is bigger than the qseq {}. {} {} {} {:?} {:?}",
                len,
                qpos,
                qseq.len(),
                window.cigar_start_offset,
                op.get_length(),
                std::str::from_utf8(cigar).unwrap(),
                range,
                window.cigar_start_idx..window.cigar_end_idx
            );
        }

        match op {
            CigarOp::Match(_) => {
                for j in 0..len {
                    let tbase = tseq[tpos + j];
                    let qbase = qseq[qpos + j];

                    if tbase == qbase {
                        m += 1;
                    } else {
                        s += 1;
                    }
                }

                tpos += len;
                qpos += len;
            }
            CigarOp::Mismatch(_) => unreachable!(),
            CigarOp::Insertion(_) => {
                i += len;
                qpos += len;
            }
            CigarOp::Deletion(_) => {
                d += len;
                tpos += len;
            }
        }
    }

    // Alignment accuracy criteria giving more weight to match and mismatches
    // (m as f32) / ((5*s + m + i + d) as f32)

    (m as f32) / ((m + s + i + d) as f32)

}



fn get_supported<S>(bases: &ArrayBase<S, Ix2>, module: &str) -> Vec<SupportedPos>
where
    S: Data<Elem = u8>,
{

    let mut counter: HashMap<u8, u8> = HashMap::default();
    counter.insert(b'A', 0);
    counter.insert(b'C', 0);
    counter.insert(b'G', 0);
    counter.insert(b'T', 0);
    counter.insert(b'*', 0);

    let mut supporeted = Vec::new();

    let (mut tpos, mut ins) = (-1i16, 0);
    for col in bases.axis_iter(Axis(0)) {
        if col[0] == b'*' {
            ins += 1;
        } else {
            tpos += 1;
            ins = 0;
        }

        let mut cov = 0;
        counter.iter_mut().for_each(|(_, c)| *c = 0);
        col.iter().for_each(|&b| {
            if b == b'.' {
                return;
            }

            cov += 1;

            if b == b'*' || b == b'#' {
                return; // skip indels
            }

            *counter.get_mut(&BASE_FORWARD[b as usize]).unwrap() += 1;
        });

        let n_supported = counter
            .iter()
            .fold(0u8, |acc, (_, &c)| if c >= 2 { acc + 1 } else { acc });
        if module != "consensus" && n_supported >= 2 && cov >= MIN_COV_TH {
            supporeted.push(SupportedPos::new(tpos as u16, ins));
        }
    }

    // println!("len_supported {:?}", supporeted.len());

    supporeted
}






fn output_features<P: AsRef<Path>>(
    path: P,
    window_id: u16,
    ids: &[&str],
    bases: Array2<u8>,
    quals: Array2<u8>,
    supported: impl IntoIterator<Item = SupportedPos>,
) -> Result<()> {
    let ids_path = path.as_ref().join(format!("{}.ids.txt", window_id));
    let ids_file = File::create(ids_path)?;
    let mut ids_writer = BufWriter::new(ids_file);
    for id in ids {
        writeln!(&mut ids_writer, "{}", id)?
    }

    let features_path = path.as_ref().join(format!("{}.features.npy", window_id));

    // Convert quals to u8 + stack feats
    let quals = quals.mapv(|q| q as u8);
    let features = stack![Axis(0), bases, quals];

    // Write feats
    let shape: Vec<_> = features.shape().iter().map(|&s| s as u64).collect();
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&shape)
        .writer(BufWriter::new(File::create(features_path)?))
        .begin_nd()?;
    writer.extend(features.iter())?;
    writer.finish()?;

    let supported_path = path.as_ref().join(format!("{}.supported.npy", window_id));
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .writer(BufWriter::new(File::create(supported_path)?))
        .begin_1d()?;
    writer.extend(supported)?;
    writer.finish()?;

    Ok(())
}

pub(crate) trait FeaturesOutput<'a> {
    fn init<'b>(&mut self, rid: u32, rname: &'b [u8])
    where
        'b: 'a;
    fn update(
        &mut self,
        rid: u32,
        wid: u16,
        bases: Array2<u8>,
        quals: Array2<u8>,
        qids: Vec<u32>,
        supported: Vec<SupportedPos>,
        ids: Vec<&str>,
        n_wids: u16,
    );
    fn emit(&mut self);
}

#[derive(Clone)]
pub(crate) struct FeatsGenOutput<'a, T>
where
    T: AsRef<Path> + Clone,
{
    base_path: T,
    rname: Option<&'a [u8]>,
    pbar_sender: Sender<PBarNotification>,
}

impl<T> FeatsGenOutput<'_, T>
where
    T: AsRef<Path> + Clone,
{
    pub(crate) fn new(path: T, pbar_sender: Sender<PBarNotification>) -> Self {
        Self {
            base_path: path,
            rname: None,
            pbar_sender: pbar_sender,
        }
    }
}

impl<'a, T> FeaturesOutput<'a> for FeatsGenOutput<'a, T>
where
    T: AsRef<Path> + Clone,
{
    fn init<'b>(&mut self, _rid: u32, rname: &'b [u8])
    where
        'b: 'a,
    {
        self.rname.replace(rname);
    }

    fn update(
        &mut self,
        _rid: u32,
        wid: u16,
        bases: Array2<u8>,
        quals: Array2<u8>,
        qids: Vec<u32>,
        supported: Vec<SupportedPos>,
        ids: Vec<&str>,
        _n_wids: u16,
    ) {
        let rid = std::str::from_utf8(self.rname.unwrap()).unwrap();
        let output_path = self.base_path.as_ref().join(rid);
        create_dir_all(&output_path).expect("Cannot create directory");

        output_features(&output_path, wid, &ids, bases, quals, supported.into_iter()).unwrap();
    }

    fn emit(&mut self) {
        self.pbar_sender.send(PBarNotification::Inc).unwrap();

        self.rname = None;
    }
}

pub(crate) struct CorrectOutput {
    sender: Sender<CorrectData>,
    features: Vec<WindowExample>,
    batch_size: usize,
}

impl CorrectOutput {
    pub(crate) fn new(sender: Sender<CorrectData>, batch_size: usize) -> Self {
        Self {
            sender,
            features: Vec::with_capacity(batch_size),
            batch_size: batch_size,
        }
    }

    pub(crate) fn finish(mut self) {
        if !self.features.is_empty() {
            let data = prepare_examples(self.features.drain(..), self.batch_size);
            self.sender.send(data).unwrap();
        }
        // sender is dropped here when `self` is dropped
    }
}

impl<'a> FeaturesOutput<'a> for CorrectOutput {
    fn init<'b>(&mut self, _rid: u32, _rname: &'b [u8])
    where
        'b: 'a,
    {
    }

    fn update(
        &mut self,
        rid: u32,
        wid: u16,
        bases: Array2<u8>,
        quals: Array2<u8>,
        qids: Vec<u32>,
        supported: Vec<SupportedPos>,
        ids: Vec<&str>,
        n_wids: u16,
    ) {
        self.features.push(WindowExample::new(
            rid,
            wid,
            ids.len() as u16,
            bases,
            quals,
            qids,
            supported,
            n_wids,
        ));

        if self.features.len() == self.batch_size {
            let data = prepare_examples(self.features.drain(..), self.batch_size);
            self.sender.send(data).unwrap();
        }
    }

    fn emit(&mut self) {
        let data = prepare_examples(self.features.drain(..), self.batch_size);
        self.sender.send(data).unwrap();
    }

}

#[derive(npyz::AutoSerialize, npyz::Serialize, PartialEq, Eq, Hash, Clone, Copy)]
#[derive(Debug)] 
pub(crate) struct SupportedPos {
    pub pos: u16,
    pub ins: u8,
}

impl SupportedPos {
    pub fn new(pos: u16, ins: u8) -> Self {
        SupportedPos { pos, ins }
    }
}
