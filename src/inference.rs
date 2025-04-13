use std::cmp::min;
use crossbeam_channel::{Receiver, Sender};
use itertools::Itertools;
use itertools::MinMaxResult::*;

use ndarray::{s, Array2, ArrayBase, Axis, Data, Ix2};
use std::cmp::max;
use rand::Rng;

use crate::{
    consensus::{ConsensusData, ConsensusWindow},
    features::SupportedPos,
};

pub(crate) const BASES_MAP: [u8; 128] = [
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 9, 255, 255,
    255, 255, 255, 255, 4, 255, 255, 255, 10, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 1, 255, 255, 255, 2, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 5, 255, 6, 255, 255, 255, 7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
];

// const BASES_UPPER: [u8; 10] = [b'A', b'C', b'G', b'T', b'*', b'A', b'C', b'G', b'T', b'*'];
const BASES_UPPER_COUNTER: [usize; 10] = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4];
const BASES_UPPER_COUNTER2: [u8; 10] = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4];
// const BASES_LOWER_COUNTER2: [u8; 10] = [5, 6, 7, 8, 9, 5, 6, 7, 8, 9];

#[derive(Debug)]
pub(crate) struct InferenceData {
    consensus_data: ConsensusData,
}

impl InferenceData {
    fn new(consensus_data: ConsensusData) -> Self {
        Self {
            consensus_data,
        }
    }
}

pub(crate) fn inference_worker(
    module: &str,
    input_channel: Receiver<InferenceData>,
    output_channel: Sender<ConsensusData>,
) {
    loop {
        let mut data = match input_channel.recv() {
            Ok(data) => data,
            Err(_) => break,
        };
        mec_modified(&mut data.consensus_data, module);
        output_channel.send(data.consensus_data).unwrap();
    }
}


fn random_f32_vector(size: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(0.0..1.0)).collect()
}


// Modified MEC or HALE code here!
fn mec_modified(data: &mut ConsensusData, module: &str) -> Option<Vec<u8>> {
    let corrected: Vec<u8> = Vec::new();

    let minmax = data
        .iter()
        .enumerate()
        .filter_map(|(idx, win)| if win.n_alns > 1 { Some(idx) } else { None })
        .minmax();
    let (wid_st, wid_en) = match minmax {
        NoElements => {
            return None;
        }
        OneElement(wid) => (wid, wid + 1),
        MinMax(st, en) => (st, en + 1),
    };
    
    for window in data[wid_st..wid_en].iter_mut() {
        if window.n_alns < 5 || module == "consensus" {
            window.bases_logits = Some(Vec::new());
            window.info_logits = Some(Vec::new());
            continue;
        }
        let n_rows = window.n_alns + 1;

        let bases = window.bases.slice(s![.., ..n_rows as usize]).to_owned();
        let informative_bases = filter_bases(&bases, &window.supported);
        let transposed = informative_bases.t().to_owned();


        let correction = if module == "hale" {
            // naive_modified_mec(&transposed)
            naive_modified_mec_weighted(&transposed)
        } else if module == "pih" {
            // pih: passive informative handling
            let row0 = transposed.row(0);
            row0.to_vec()
        } else {
            panic!("Invalid module name: {}", module);
        };
        
        let corr2 = correction.clone();
        window.bases_logits = Some(correction);
        window.info_logits = Some(random_f32_vector(corr2.len()));

    }

    Some(corrected)
}


// get informative sites matrix
fn filter_bases(bases: &Array2<u8>, supported: &[SupportedPos]) -> Array2<u8> {
    let filtered_indices: Vec<usize> = bases
    .axis_iter(Axis(0))  
    .enumerate()   
    .filter(|(_, row)| (row[0] != 4 && row[0] != 9)) 
    .map(|(idx, _)| idx) 
    .collect();

    let supported_map: Vec<(usize, usize)> = supported
    .iter()
    .map(|s| (s.pos as usize, s.ins as usize))
    .collect();

    let mut selected_rows: Vec<Vec<u8>> = Vec::new();

    for &(orig_idx, ins) in &supported_map {
        if let Some(&filtered_idx) = filtered_indices.get(orig_idx) {
            let target_idx = filtered_idx + ins;
            selected_rows.push(bases.row(target_idx).to_vec());
        }
    }

    let num_cols = bases.ncols();
    let filtered_bases = Array2::<u8>::from_shape_fn((selected_rows.len(), num_cols), |(i, j)| {
        selected_rows[i][j]
    });

    filtered_bases
}


/// Computes the cost for a given bitmask
fn get_bitmask_cost(bitmask: u32, bases: &Array2<u8>, set_bits: u32) -> u32 {
    let n = bases.nrows();
    let m = bases.ncols();
    let mut cost = 0;

    for i in 0..m {
        let mut a_base = (bases[[0, i]] == BASES_MAP[b'A' as usize] || bases[[0, i]] == BASES_MAP[b'a' as usize]) as u32;
        let mut t_base = (bases[[0, i]] == BASES_MAP[b'T' as usize] || bases[[0, i]] == BASES_MAP[b't' as usize]) as u32;
        let mut c_base = (bases[[0, i]] == BASES_MAP[b'C' as usize] || bases[[0, i]] == BASES_MAP[b'c' as usize]) as u32;
        let mut g_base = (bases[[0, i]] == BASES_MAP[b'G' as usize] || bases[[0, i]] == BASES_MAP[b'g' as usize]) as u32;
        let mut d_base = (bases[[0, i]] == BASES_MAP[b'*' as usize] || bases[[0, i]] == BASES_MAP[b'#' as usize]) as u32;

        for j in 1..n {
            if (bitmask & (1 << (j - 1))) != 0 {
                let base = bases[[j, i]];

                if base != BASES_MAP[b'.' as usize] {
                    match base {
                        x if x == BASES_MAP[b'A' as usize] || x == BASES_MAP[b'a' as usize] => a_base += 1,
                        x if x == BASES_MAP[b'T' as usize] || x == BASES_MAP[b't' as usize] => t_base += 1,
                        x if x == BASES_MAP[b'C' as usize] || x == BASES_MAP[b'c' as usize] => c_base += 1,
                        x if x == BASES_MAP[b'G' as usize] || x == BASES_MAP[b'g' as usize] => g_base += 1,
                        x if x == BASES_MAP[b'*' as usize] || x == BASES_MAP[b'#' as usize] => d_base += 1,
                        _ => (),
                    }
                }
            }
        }

        cost += set_bits + 1 - max(max(a_base, c_base), max(max(g_base, t_base), d_base));

    }
    
    cost
}


/// Computes the cost for a given bitmask multiplied by weight of column
fn get_bitmask_cost_weighted(bitmask: u32, bases: &Array2<u8>, set_bits: u32, weights: &Vec<f32>) -> f32 {
    let n = bases.nrows();
    let m = bases.ncols();
    let mut cost = 0 as f32;

    for i in 0..m {
        let mut a_base = (bases[[0, i]] == BASES_MAP[b'A' as usize] || bases[[0, i]] == BASES_MAP[b'a' as usize]) as u32;
        let mut t_base = (bases[[0, i]] == BASES_MAP[b'T' as usize] || bases[[0, i]] == BASES_MAP[b't' as usize]) as u32;
        let mut c_base = (bases[[0, i]] == BASES_MAP[b'C' as usize] || bases[[0, i]] == BASES_MAP[b'c' as usize]) as u32;
        let mut g_base = (bases[[0, i]] == BASES_MAP[b'G' as usize] || bases[[0, i]] == BASES_MAP[b'g' as usize]) as u32;
        let mut d_base = (bases[[0, i]] == BASES_MAP[b'*' as usize] || bases[[0, i]] == BASES_MAP[b'#' as usize]) as u32;

        for j in 1..n {
            if (bitmask & (1 << (j - 1))) != 0 {
                let base = bases[[j, i]];
                let base_0 = bases[[0, i]];

                if base != BASES_MAP[b'.' as usize] && base_0 != BASES_MAP[b'.' as usize] {
                    match base {
                        x if x == BASES_MAP[b'A' as usize] || x == BASES_MAP[b'a' as usize] => a_base += 1,
                        x if x == BASES_MAP[b'T' as usize] || x == BASES_MAP[b't' as usize] => t_base += 1,
                        x if x == BASES_MAP[b'C' as usize] || x == BASES_MAP[b'c' as usize] => c_base += 1,
                        x if x == BASES_MAP[b'G' as usize] || x == BASES_MAP[b'g' as usize] => g_base += 1,
                        x if x == BASES_MAP[b'*' as usize] || x == BASES_MAP[b'#' as usize] => d_base += 1,
                        _ => (),
                    }
                }
            }
        }
        

        cost += weights[i]*(set_bits + 1 - max(max(a_base, c_base), max(max(g_base, t_base), d_base))) as f32;

    }
    
    cost
}


/// Finds the optimal correction sequence using the best partition
fn naive_modified_mec(bases: &Array2<u8>) -> Vec<u8> {
    let n = bases.nrows();
    let m = bases.ncols();

    let total_bits = (n - 1) as u32;
    let mut set_bits = max(2, total_bits / 3);

    let mut bitmask = (1 << set_bits) - 1; // Smallest bitmask with `set_bits` bits set
    let mut min_cost = u32::MAX;
    let mut corr_mask = 0;

    // Iterate over all valid bitmasks
    while bitmask < (1 << total_bits) {
        let cost = get_bitmask_cost(bitmask, bases, set_bits);
        if cost < min_cost {
            min_cost = cost;
            corr_mask = bitmask;
        }

        // Generate the next bitmask with the same number of set bits
        let t = bitmask | (bitmask - 1);
        bitmask = (t + 1) | (((!t & (!t).wrapping_neg()) - 1) >> (bitmask.trailing_zeros() + 1));
    }

    // Compute the corrected sequence
    let mut corrections: Vec<u8> = vec![4; m]; // Default to '*' if no match

    for i in 0..m {
        let mut counts: Vec<u8> = vec![0; 5]; // Assuming 5 possible bases (A, T, C, G, '*')
    
        for j in 0..n {
            if j == 0 || (corr_mask & (1 << (j - 1))) != 0 {
                let base = bases[[j, i]];
                if base != BASES_MAP[b'.' as usize] {
                    counts[BASES_UPPER_COUNTER[base as usize]] += 1;
                }
            }
        }

        if let Some(max_count) = counts.iter().max() {
            let mut best_index = None;
            let mut tie = false;
        
            for (idx, &count) in counts.iter().enumerate() {
                if count == *max_count {
                    if best_index.is_some() {
                        tie = true;
                    }
                    best_index = Some(idx);
                }
            }
        
            if let Some(best_idx) = best_index {
                if tie {
                    let target_base_idx = BASES_UPPER_COUNTER[bases[[0, i]] as usize];
                    if counts[target_base_idx] == *max_count {
                        best_index = Some(target_base_idx);
                    }
                }
                corrections[i] = BASES_UPPER_COUNTER2[best_index.unwrap()];
                // corrections[i] = BASES_LOWER_COUNTER2[best_index.unwrap()];
            }

        }
        
    }

    corrections
}


fn get_col_weight(bases: &Array2<u8>) -> Vec<f32> {
    let n = bases.nrows();
    let m = bases.ncols();
    let mut weights = vec![1f32; m];

    let mut base_counts = [0u32; 5]; // [A, C, G, T, D]
    for i in 0..m {
        base_counts.fill(0);

        for j in 0..n {
            let base = bases[[j, i]];
            if base == BASES_MAP[b'.' as usize] {
                continue;
            }
            match base {
                x if x == BASES_MAP[b'A' as usize] || x == BASES_MAP[b'a' as usize] => base_counts[0] += 1,
                x if x == BASES_MAP[b'C' as usize] || x == BASES_MAP[b'c' as usize] => base_counts[1] += 1,
                x if x == BASES_MAP[b'G' as usize] || x == BASES_MAP[b'g' as usize] => base_counts[2] += 1,
                x if x == BASES_MAP[b'T' as usize] || x == BASES_MAP[b't' as usize] => base_counts[3] += 1,
                x if x == BASES_MAP[b'*' as usize] || x == BASES_MAP[b'#' as usize] => base_counts[4] += 1,
                _ => (),
            }
        }

        let mut counts = base_counts.clone();
        counts.sort_unstable_by(|a, b| b.cmp(a));
        let max_base = counts[0];
        let second_max = counts[1];
        if max_base - second_max < 5 {
            weights[i] = 4 as f32;
        } else if max_base - second_max < 10 {
            weights[i] = 2 as f32;
        } else {
            weights[i] = 1 as f32;
        }

        // weights[i] = 1 as f32;

        if base_counts[4]==max_base || bases[[0,i]] == BASES_MAP[b'*' as usize] || bases[[0,i]] == BASES_MAP[b'#' as usize] {
            weights[i] = 0.1 as f32;
        }

        // weights[i] = ((max_base + second_max) as f32)/ ((max_base - second_max + 1) as f32);

    }

    weights
}


/// Finds the optimal correction sequence using the best partition
fn naive_modified_mec_weighted(bases: &Array2<u8>) -> Vec<u8> {
    let n = bases.nrows();
    let m = bases.ncols();

    let total_bits = (n - 1) as u32;
    let mut set_bits = max(2, total_bits / 3);
    
    let mut bitmask = (1 << set_bits) - 1; // Smallest bitmask with `set_bits` bits set
    let mut min_cost = f32::MAX;
    let mut corr_mask = 0;
    let weights = get_col_weight(bases);


    // Iterate over all valid bitmasks
    while bitmask < (1 << total_bits) {
        let cost = get_bitmask_cost_weighted(bitmask, bases, set_bits, &weights);
        if cost < min_cost {
            min_cost = cost;
            corr_mask = bitmask;
        }

        // Generate the next bitmask with the same number of set bits
        let t = bitmask | (bitmask - 1);
        bitmask = (t + 1) | (((!t & (!t).wrapping_neg()) - 1) >> (bitmask.trailing_zeros() + 1));
    }

    // Compute the corrected sequence
    let mut corrections: Vec<u8> = vec![4; m]; // Default to '*' if no match

    for i in 0..m {
        let mut counts: Vec<u8> = vec![0; 5]; // Assuming 5 possible bases (A, T, C, G, '*')
    
        for j in 0..n {
            if j == 0 || (corr_mask & (1 << (j - 1))) != 0 {
                let base = bases[[j, i]];
                if base != BASES_MAP[b'.' as usize] {
                    counts[BASES_UPPER_COUNTER[base as usize]] += 1;
                }
            }
        }

        if let Some(max_count) = counts.iter().max() {
            let mut best_index = None;
            let mut tie = false;
        
            for (idx, &count) in counts.iter().enumerate() {
                if count == *max_count {
                    if best_index.is_some() {
                        tie = true;
                    }
                    best_index = Some(idx);
                }
            }
        
            if let Some(best_idx) = best_index {
                if tie {
                    let target_base_idx = BASES_UPPER_COUNTER[bases[[0, i]] as usize];
                    if counts[target_base_idx] == *max_count {
                        best_index = Some(target_base_idx);
                    }
                }
                corrections[i] = BASES_UPPER_COUNTER2[best_index.unwrap()];
                // corrections[i] = BASES_LOWER_COUNTER2[best_index.unwrap()];
            }

        }
        
    }

    corrections
}




pub(crate) fn prepare_examples(
    features: impl IntoIterator<Item = WindowExample>,
    batch_size: usize,
) -> InferenceData {
    let windows: Vec<_> = features
        .into_iter()
        .map(|mut example| {
            // Transform bases (encode) and quals (normalize)
            example.bases.mapv_inplace(|b| BASES_MAP[b as usize]);

            // Transpose: [R, L] -> [L, R]
            //bases.swap_axes(1, 0);
            //quals.swap_axes(1, 0);

            let tidx = get_target_indices(&example.bases);

            //TODO: Start here.
            ConsensusWindow::new(
                example.rid,
                example.wid,
                example.n_alns,
                example.n_total_wins,
                example.bases,
                example.quals,
                example.qids,
                tidx,
                example.supported,
                None,
                None,
            )
        })
        .collect();

    // let batches: Vec<_> = (0u32..)
    //     .zip(windows.iter())
    //     .filter(|(_, features)| features.supported.len() > 0)
    //     .chunks(batch_size)
    //     .into_iter()
    //     .map(|v| {
    //         let batch = v.collect::<Vec<_>>();
    //         collate(&batch)
    //     })
    //     .collect();

    InferenceData::new(windows)
}

fn get_target_indices<S: Data<Elem = u8>>(bases: &ArrayBase<S, Ix2>) -> Vec<usize> {
    bases
        .slice(s![.., 0])
        .iter()
        .enumerate()
        .filter_map(|(idx, b)| {
            if *b != BASES_MAP[b'*' as usize] {
                Some(idx)
            } else {
                None
            }
        })
        .collect()
}

pub(crate) struct WindowExample {
    rid: u32,
    wid: u16,
    n_alns: u8,
    bases: Array2<u8>,
    quals: Array2<u8>,
    qids: Vec<u32>,
    supported: Vec<SupportedPos>,
    n_total_wins: u16,
}

impl WindowExample {
    pub(crate) fn new(
        rid: u32,
        wid: u16,
        n_alns: u8,
        bases: Array2<u8>,
        quals: Array2<u8>,
        qids: Vec<u32>,
        supported: Vec<SupportedPos>,
        n_total_wins: u16,
    ) -> Self {
        Self {
            rid,
            wid,
            n_alns,
            bases,
            quals,
            qids,
            supported,
            n_total_wins,
        }
    }
}

