// use std::path::Path;
use std::cmp::min;
use crossbeam_channel::{Receiver, Sender};
use itertools::Itertools;
use itertools::MinMaxResult::*;

use std::collections::HashMap;
use ndarray::{s, Array2, ArrayBase, Axis, Data, Ix2};
use std::collections::HashSet;
use std::cmp::max;
use ndarray::concatenate;
// use tch::{CModule, IValue, IndexOp, Tensor};
use rand::Rng;

use crate::{
    consensus::{self, ConsensusData, ConsensusWindow},
    features::SupportedPos,
};

// const BASE_PADDING: u8 = 11;
// const QUAL_MIN_VAL: f32 = 33.;
// const QUAL_MAX_VAL: f32 = 126.;

// const QUAL_RANGE_DIFF: f64 = (QUAL_MAX_VAL - QUAL_MIN_VAL) as f64;
// const QUAL_SCALE: f64 = 2. / QUAL_RANGE_DIFF;
// const QUAL_OFFSET: f64 = 2. * QUAL_MIN_VAL as f64 / QUAL_RANGE_DIFF + 1.;

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

// #[derive(Debug)]
// pub(crate) struct InferenceBatch {
//     wids: Vec<u32>,
//     bases: Tensor,
//     quals: Tensor,
//     lens: Tensor,
//     indices: Vec<Tensor>,
// }

// impl InferenceBatch {
//     fn new(
//         wids: Vec<u32>,
//         bases: Tensor,
//         quals: Tensor,
//         lens: Tensor,
//         indices: Vec<Tensor>,
//     ) -> Self {
//         Self {
//             wids,
//             bases,
//             quals,
//             lens,
//             indices,
//         }
//     }
// }

#[derive(Debug)]
pub(crate) struct InferenceData {
    consensus_data: ConsensusData,
    // batches: Vec<InferenceBatch>,
}

impl InferenceData {
    fn new(consensus_data: ConsensusData) -> Self {
        Self {
            consensus_data,
            // batches,
        }
    }
}

// fn collate<'a>(batch: &[(u32, &ConsensusWindow)]) -> InferenceBatch {
//     // Get longest sequence
//     let length = batch
//         .iter()
//         .map(|(_, f)| f.bases.len_of(Axis(0)))
//         .max()
//         .unwrap();
//     let size = [
//         batch.len() as i64,
//         length as i64,
//         batch[0].1.bases.len_of(Axis(1)) as i64,
//     ]; // [B, L, R]

//     let bases = Tensor::full(
//         &size,
//         BASE_PADDING as i64,
//         (tch::Kind::Uint8, tch::Device::Cpu),
//     );

//     //let quals = Tensor::ones(&size, (tch::Kind::Uint8, tch::Device::Cpu));
//     let quals = Tensor::full(
//         &size,
//         QUAL_MAX_VAL as i64,
//         (tch::Kind::Uint8, tch::Device::Cpu),
//     );

//     let mut lens = Vec::with_capacity(batch.len());
//     let mut indices = Vec::with_capacity(batch.len());
//     let mut wids = Vec::with_capacity(batch.len());

//     for (idx, (wid, f)) in batch.iter().enumerate() {
//         wids.push(*wid);
//         let l = f.bases.len_of(Axis(0));

//         let bt = unsafe {
//             let shape: Vec<_> = f.bases.shape().iter().map(|s| *s as i64).collect();
//             Tensor::from_blob(
//                 f.bases.as_ptr(),
//                 &shape,
//                 &[shape[shape.len() - 1], 1],
//                 tch::Kind::Uint8,
//                 tch::Device::Cpu,
//             )
//         };

//         let qt = unsafe {
//             let shape: Vec<_> = f.quals.shape().iter().map(|s| *s as i64).collect();
//             Tensor::from_blob(
//                 f.quals.as_ptr() as *const u8,
//                 &shape,
//                 &[shape[shape.len() - 1], 1],
//                 tch::Kind::Uint8,
//                 tch::Device::Cpu,
//             )
//         };

//         bases.i((idx as i64, ..l as i64, ..)).copy_(&bt);
//         quals.i((idx as i64, ..l as i64, ..)).copy_(&qt);

//         /*for p in 0..l {
//             for r in 0..f.bases.len_of(Axis(1)) {
//                 let _ = bases
//                     .i((idx as i64, p as i64, r as i64))
//                     .fill_(f.bases[[p, r]] as i64);

//                 let _ = quals
//                     .i((idx as i64, p as i64, r as i64))
//                     .fill_(f.quals[[p, r]] as f64);
//             }
//         }*/

//         //println!("Bases shape: {:?}", f.bases.shape());
//         //println!("Quals shape: {:?}", f.quals.shape());

//         lens.push(f.supported.len() as i32);

//         let tidx: Vec<_> = f
//             .supported
//             .iter()
//             .map(|&sp| (f.indices[sp.pos as usize] + sp.ins as usize) as i32)
//             .collect();
//         indices.push(Tensor::try_from(tidx).unwrap());
//     }

//     /*if batch[0].1.supported.contains(&SupportedPos::new(837, 0))
//         && batch[0].1.supported.contains(&SupportedPos::new(1157, 0))
//     {
//         bases.save("bases_to_test.tmp2.pt").unwrap();
//         quals.save("quals_to_test.tmp2.pt").unwrap();
//         indices[0].save("indices_to_test.tmp2.pt").unwrap();
//     }*/

//     InferenceBatch::new(wids, bases, quals, Tensor::try_from(lens).unwrap(), indices)
// }

// fn inference(
//     batch: InferenceBatch,
//     model: &CModule,
//     device: tch::Device,
// ) -> (Vec<u32>, Vec<Tensor>, Vec<Tensor>) {
//     let quals = batch.quals.to_device_(device, tch::Kind::Float, true, true);
//     let quals = QUAL_SCALE * quals - QUAL_OFFSET;

//     let inputs = [
//         IValue::Tensor(batch.bases.to_device_(device, tch::Kind::Int, true, true)),
//         IValue::Tensor(quals),
//         IValue::Tensor(batch.lens),
//         IValue::TensorList(batch.indices),
//     ];

//     let (info_logits, bases_logits) =
//         <(Tensor, Tensor)>::try_from(model.forward_is(&inputs).unwrap()).unwrap();

//     // Get number of target positions for each window
//     let lens: Vec<i64> = match inputs[2] {
//         IValue::Tensor(ref t) => Vec::try_from(t).unwrap(),
//         _ => unreachable!(),
//     };

//     let info_logits = info_logits.to(tch::Device::Cpu).split_with_sizes(&lens, 0);
//     let bases_logits = bases_logits
//         .argmax(1, false)
//         .to(tch::Device::Cpu)
//         .split_with_sizes(&lens, 0);

//     (batch.wids, info_logits, bases_logits)
// }


pub(crate) fn inference_worker(
    // model_path: P,
    // device: tch::Device,
    module: &str,
    input_channel: Receiver<InferenceData>,
    output_channel: Sender<ConsensusData>,
) {
    // let _no_grad = tch::no_grad_guard();

    // let mut model = tch::CModule::load_on_device(model_path, device).expect("Cannot load model.");
    // model.set_eval();

    loop {
        let mut data = match input_channel.recv() {
            Ok(data) => data,
            Err(_) => break,
        };

        // mec_modified2(&mut data.consensus_data);
        mec_modified(&mut data.consensus_data, module);

        output_channel.send(data.consensus_data).unwrap();
    }
}


fn random_f32_vector(size: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(0.0..1.0)).collect()
}


fn count_unique_rows(bases: &Array2<u8>) -> usize {
    let mut unique_rows = HashSet::new();

    for row in bases.rows() {
        unique_rows.insert(row.to_vec()); // Convert row to Vec<u8> for hashing
    }

    unique_rows.len()
}


// MEC code here!
fn mec_modified(data: &mut ConsensusData, module: &str) -> Option<Vec<u8>> {

    let mut corrected: Vec<u8> = Vec::new();

    // Picking code from consensus that will fit here as well:
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

    // println!("Windows and respective informative bases:");

    // let rids_of_interest = vec![ 352392, 41134, 144192, 144197, 144181, 144232, 144184, 247209, 247213,247166, 244047, 350192, 350120, 350194];
    // let rids_of_interest = vec![ 352392, 350120, 350194];


    for window in data[wid_st..wid_en].iter_mut() {
        if window.n_alns < 6 || module == "consensus" {
            window.bases_logits = Some(Vec::new());
            window.info_logits = Some(Vec::new());
            continue;
        }
        let n_rows = min(20, window.n_alns) + 1;

        // let full_bases = window.bases.to_owned();
        // let informative_bases_full = filter_bases(&full_bases, &window.supported);
        // let transposed_full = informative_bases_full.t().to_owned();

        let bases = window.bases.slice(s![.., ..n_rows as usize]).to_owned();
        // I also have to select only the rows that arre supported!
        let informative_bases = filter_bases_2(&bases, &window.supported);
        // Now we need to do MEC on this window!
        // get the corrected bases as supported positions and then update those bases as info_logits of that window
        // Note: I have already removed indels from the definition of supported position
        let transposed = informative_bases.t().to_owned();

        // let count = count_unique_rows(&transposed);
        // let total_rows = transposed.nrows();
        // let total_cols = transposed.ncols();
        // if total_cols > 3 {
        //     println!("Unique rows vs toatl rows: {} {}", count, total_rows);
        // }


        let correction = if module == "hale" {
            naive_modified_mec(&transposed)
        } else if module == "binary_mec" {
            naive_modified_mec_original(&transposed)
        } else {
            panic!("Invalid module name: {}", module);
        };
        
        
        // println!("size of the matrix before transpose: {} {}", informative_bases.nrows(), informative_bases.ncols());
        // println!("size of the matrix to pass: {} {}", transposed.nrows(), transposed.ncols());

        // println!("Corrections in this window are: {:#?}", correction);

        // println!("matrix::\n {:#?}", transposed);
        // println!("window logits before: {:#?}", window.bases_logits);

        // if(window.rid == 251){ println!("window before :\n {:#?}", window); }

        // if rids_of_interest.contains(&window.rid) {
        //     println!("rid: {:#?} \n 20 rows used for MEC : \n {:#?}", window.rid, transposed);
        //     println!("window base logits before :\n {:#?}", window.bases_logits);
        // }

        // println!("rid: {:#?} \n MEC Mtarix rows x cols: {:#?} x  {:#?}", window.rid, transposed.nrows(), transposed.ncols());

        
        let corr2 = correction.clone(); // Use `.clone()` for a deep copy
        window.bases_logits = Some(correction);
        window.info_logits = Some(random_f32_vector(corr2.len()));

        // if(window.rid == 251){ println!("window after :\n {:#?}", window); }
        // println!("bases: {:#?}", informative_bases);

        // if rids_of_interest.contains(&window.rid) {
        //     println!("window base logits after :\n {:#?}", window.bases_logits);
        // }
    }


    Some(corrected)
}


// MEC code here!
fn mec_modified2(data: &mut ConsensusData) -> Option<Vec<u8>> {

    let mut corrected: Vec<u8> = Vec::new();

    // Picking code from consensus that will fit here as well:
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

    // println!("Windows and respective informative bases:");

    // let rids_of_interest = vec![61535,61510,2670,22442,22473,42049,2601,2558,61394,22322,22366,2584,61436,22409,22375];

    let mut top_20_map: HashMap<u32, u32> = HashMap::new();
    let mut top_6_map: HashMap<u32, u32> = HashMap::new();

    for window in data[wid_st..wid_en].iter_mut() {
        if window.n_alns < 6 {
            continue;
        }
        let mut n_rows: usize = (std::cmp::min(20, window.n_alns) + 1) as usize;
        let mut n_rows_2: usize = 0;

        if(window.n_alns > 18) {
            n_rows = (1 + window.n_alns/2) as usize;
            n_rows_2 = (1 + (1+window.n_alns)/2) as usize;
        }

        if n_rows_2 > 0 {

            let bases1 = window.bases.slice(s![.., ..n_rows as usize]).to_owned();
            let first_row = window.bases.slice(s![.., 0..1]);
            let remaining_rows = window.bases.slice(s![.., n_rows-1 as usize..]); 
            let bases2 = concatenate(Axis(1), &[first_row, remaining_rows]).unwrap();

            let qids = window.qids.to_owned();
            // I also have to select only the rows that arre supported!
            let informative_bases1 = filter_bases(&bases1, &window.supported);
            let informative_bases2 = filter_bases(&bases2, &window.supported);
            // Now we need to do MEC on this window!
            // get the corrected bases as supported positions and then update those bases as info_logits of that window
            // Note: I have already removed indels from the definition of supported position
            let transposed1 = informative_bases1.t().to_owned();
            let transposed2 = informative_bases2.t().to_owned();

            // println!("transposed1: {:#?}", transposed1);
            // println!("transposed2: {:#?}", transposed2);

            let corr_mask1 = naive_modified_mec2(&transposed1);
            let corr_mask2 = naive_modified_mec2(&transposed2);

            // Now I can know the reads that were selcted by MEC in this wondow
            // I will save the number of times that read was in transposed matrix i.e. in top 20 and number of times it was in corr mask
            // Then i will do consenus outside this loop.

            for i in 0..n_rows-1 {
                if (1 << (i)) & corr_mask1 != 0 {
                    *top_6_map.entry(qids[i]).or_insert(0) += 1;
                }
                *top_20_map.entry(qids[i]).or_insert(0) += 1;
            }

            for i in 0..n_rows_2-1 {
                if (1 << (i)) & corr_mask2 != 0 {
                    *top_6_map.entry(qids[n_rows-1+i]).or_insert(0) += 1;
                }
                *top_20_map.entry(qids[n_rows-1+i]).or_insert(0) += 1;
            }

        } else {

            // let full_bases = window.bases.to_owned();
            // let informative_bases_full = filter_bases(&full_bases, &window.supported);
            // let transposed_full = informative_bases_full.t().to_owned();

            let bases = window.bases.slice(s![.., ..n_rows as usize]).to_owned();
            let qids = window.qids.to_owned();
            // I also have to select only the rows that arre supported!
            let informative_bases = filter_bases(&bases, &window.supported);
            // Now we need to do MEC on this window!
            // get the corrected bases as supported positions and then update those bases as info_logits of that window
            // Note: I have already removed indels from the definition of supported position
            let transposed = informative_bases.t().to_owned();

            let corr_mask = naive_modified_mec2(&transposed);

            // Now I can know the reads that were selcted by MEC in this wondow
            // I will save the number of times that read was in transposed matrix i.e. in top 20 and number of times it was in corr mask
            // Then i will do consenus outside this loop.

            for i in 0..std::cmp::min(20, qids.len()) {
                if (1 << (i)) & corr_mask != 0 {
                    *top_6_map.entry(qids[i]).or_insert(0) += 1;
                }
                *top_20_map.entry(qids[i]).or_insert(0) += 1;
            }
        }
        
    }


    // this time, we will do consensus for correcting bases art informative sites
    for window in data[wid_st..wid_en].iter_mut() {
        if window.n_alns < 6 {
            continue;
        }
        let mut n_rows = min(20, window.n_alns) + 1;
        let mut n_rows_2: u8 = 0;

        if(window.n_alns > 18) {
            n_rows = 1 + window.n_alns/2;
            n_rows_2 = 1 + (1+window.n_alns)/2;
        }

        let mut bases = window.bases.slice(s![.., ..n_rows as usize]).to_owned();
        if n_rows_2 > 0 {
            bases = window.bases.slice(s![.., ..(window.n_alns+1) as usize]).to_owned();
        }
        let qids = window.qids.to_owned();
        
        let informative_bases = filter_bases(&bases, &window.supported);
        let transposed = informative_bases.t().to_owned();
        
        let n = transposed.nrows();
        let m = transposed.ncols();
        // println!("rows, cols: {:#?} {:#?} {:#?}", n, m, qids.len());

        if m>0 {
            let mut corrections: Vec<u8> = vec![4; m];
            for i in 0..m {
                let mut counts: Vec<f32> = vec![0.0; 5]; 
                
                for j in 0..n {
                    let mut prop: f32 = 0.0;
                    if j == 0 {
                        prop = 1.0;
                    } else if j > 0 && top_20_map.contains_key(&qids[j - 1]) {
                        let top_6_val = *top_6_map.get(&qids[j-1]).unwrap_or(&0) as f32;
                        let top_20_val = *top_20_map.get(&qids[j - 1]).unwrap() as f32;
                        if top_20_val != 0.0 {
                            prop = top_6_val / top_20_val;
                        }
                    }
        
                    let base = transposed[[j, i]]; // Correct indexing
                    if base != BASES_MAP[b'.' as usize] {
                        counts[BASES_UPPER_COUNTER[base as usize]] += prop;
                    }
                }

                // println!("weighted counts: {:#?}", counts);
        
                if let Some(max_count) = counts.iter().cloned().max_by(|a, b| a.partial_cmp(b).unwrap()) {
                    let mut best_index = None;
                    let mut tie = false;
        
                    for (idx, &count) in counts.iter().enumerate() {
                        if count == max_count {
                            if best_index.is_some() {
                                tie = true;
                            }
                            best_index = Some(idx);
                        }
                    }
        
                    if let Some(best_idx) = best_index {
                        if tie {
                            let target_base_idx = BASES_UPPER_COUNTER[transposed[[0, i]] as usize];
                            if counts[target_base_idx] == max_count {
                                best_index = Some(target_base_idx);
                            }
                        }
                        corrections[i] = BASES_UPPER_COUNTER2[best_index.unwrap()];
                    }
                }
            
            }

            window.bases_logits = Some(corrections);
        }
    }


    Some(corrected)
}



fn filter_bases(bases: &Array2<u8>, supported: &[SupportedPos]) -> Array2<u8> {
    // Step 1: Filter rows where bases[row][0] is not 4
    let filtered_rows: Vec<Vec<u8>> = bases
        .axis_iter(Axis(0))
        .filter(|row| row[0] != 4)
        .map(|row| row.to_vec()) // Convert to Vec<u8> for indexing
        .collect();

    // Step 2: Convert `supported.pos` into a HashSet for fast lookups
    let supported_set: HashSet<u16> = supported.iter().map(|s| s.pos).collect();

    // Step 3: Keep only those rows whose new indices (after filtering) are in `supported_set`
    let final_rows: Vec<Vec<u8>> = filtered_rows
        .iter()
        .enumerate()
        .filter(|(new_idx, _)| supported_set.contains(&(*new_idx as u16))) // Using the new index
        .map(|(_, row)| row.clone()) // Clone the filtered row
        .collect();

    // Step 4: Create a new filtered matrix
    let num_cols = bases.ncols();
    let filtered_bases = Array2::<u8>::from_shape_fn((final_rows.len(), num_cols), |(i, j)| {
        final_rows[i][j]
    });

    filtered_bases
}



fn filter_bases_2(bases: &Array2<u8>, supported: &[SupportedPos]) -> Array2<u8> {
    // Step 1: Filter rows where bases[row][0] is not 4
    let filtered_indices: Vec<usize> = bases
    .axis_iter(Axis(0))  // Iterate over rows
    .enumerate()          // Get (index, row)
    .filter(|(_, row)| row[0] != 4) // Keep only rows where row[0] != 4
    .map(|(idx, _)| idx)  // Extract only the index
    .collect();

    let supported_map: Vec<(usize, usize)> = supported
    .iter()
    .map(|s| (s.pos as usize, s.ins as usize))
    .collect();

    let mut selected_rows: Vec<Vec<u8>> = Vec::new();

    for &(orig_idx, ins) in &supported_map {
        if let Some(&filtered_idx) = filtered_indices.get(orig_idx) {
            let target_idx = filtered_idx + ins; // Ensure `ins` is the correct type (usize or i32)
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

    // println!("bases {:#?}", bases);
    // println!("size of bases: Rows x Cols {} {}", n, m);

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

        cost += set_bits + 1 - max(max(a_base, c_base), max(max(g_base, t_base), d_base));

    }
    
    cost
}

/// Computes the cost for a given bitmask old one
fn get_bitmask_cost_old(bitmask: u32, bases: &Array2<u8>) -> u32 {
    let n = bases.nrows();
    let m = bases.ncols();
    let mut cost = 0;

    // println!("bases {:#?}", bases);
    // println!("size of bases: Rows x Cols {} {}", n, m);

    for i in 0..m {
        let mut ones = 1;
        let mut zeroes = 0;

        for j in 1..n {
            if (bitmask & (1 << (j - 1))) != 0 {
                if bases[[j, i]] != BASES_MAP[b'.' as usize] && bases[[0, i]] != BASES_MAP[b'.' as usize] {
                    if(BASES_UPPER_COUNTER[bases[[j, i]] as usize] == BASES_UPPER_COUNTER[bases[[0, i]] as usize]) {ones += 1;}
                    else {zeroes += 1;}
                }
            }
        }
        cost += min(ones, zeroes);

    }
    cost
}



/// Computes the cost for a given bitmask
fn get_bitmask_cost_original(bitmask: u32, bases: &Array2<u8>) -> u32 {
    let n = bases.nrows();
    let m = bases.ncols();
    let mut cost = 0;

    for i in 0..m {
        let (mut a0_base, mut t0_base, mut c0_base, mut g0_base, mut d0_base) = (0u32, 0u32, 0u32, 0u32, 0u32);
        let (mut a1_base, mut t1_base, mut c1_base, mut g1_base, mut d1_base) = (0u32, 0u32, 0u32, 0u32, 0u32);

        for j in 0..n {
            let base = bases[[j, i]];
            if (bitmask & (1 << j)) != 0 {
                if base != BASES_MAP[b'.' as usize] {
                    match base {
                        x if x == BASES_MAP[b'A' as usize] || x == BASES_MAP[b'a' as usize] => a0_base += 1,
                        x if x == BASES_MAP[b'T' as usize] || x == BASES_MAP[b't' as usize] => t0_base += 1,
                        x if x == BASES_MAP[b'C' as usize] || x == BASES_MAP[b'c' as usize] => c0_base += 1,
                        x if x == BASES_MAP[b'G' as usize] || x == BASES_MAP[b'g' as usize] => g0_base += 1,
                        x if x == BASES_MAP[b'*' as usize] || x == BASES_MAP[b'#' as usize] => d0_base += 1,
                        _ => (),
                    }
                }
            } else {
                if base != BASES_MAP[b'.' as usize] {
                    match base {
                        x if x == BASES_MAP[b'A' as usize] || x == BASES_MAP[b'a' as usize] => a1_base += 1,
                        x if x == BASES_MAP[b'T' as usize] || x == BASES_MAP[b't' as usize] => t1_base += 1,
                        x if x == BASES_MAP[b'C' as usize] || x == BASES_MAP[b'c' as usize] => c1_base += 1,
                        x if x == BASES_MAP[b'G' as usize] || x == BASES_MAP[b'g' as usize] => g1_base += 1,
                        x if x == BASES_MAP[b'*' as usize] || x == BASES_MAP[b'#' as usize] => d1_base += 1,
                        _ => (),
                    }
                }
            }
            let part0_bases = [a0_base, c0_base, g0_base, d0_base, t0_base];
            cost += part0_bases.iter().sum::<u32>() - *part0_bases.iter().max().unwrap();
            let part1_bases = [a1_base, c1_base, g1_base, d1_base, t1_base];
            cost += part1_bases.iter().sum::<u32>() - *part1_bases.iter().max().unwrap();
            
        }
    }
    cost
}



/// Finds the optimal correction sequence using the best partition
fn naive_modified_mec(bases: &Array2<u8>) -> Vec<u8> {
    let n = bases.nrows();
    let m = bases.ncols();

    // println!("bases: {:#?}", bases);
    // println!("size of the matrix: {} {}", n, m);

    let total_bits = (n - 1) as u32;
    let mut set_bits = min(7, total_bits / 4);
    if total_bits < 24{
        set_bits = min(6, total_bits / 3); // max 6 rows, or n/3
    } 
    // let set_bits = max(2, total_bits+3 / 5); // max 6 rows, or n/3
    let mut bitmask = (1 << set_bits) - 1; // Smallest bitmask with `set_bits` bits set

    let mut min_cost = u32::MAX;
    let mut corr_mask = 0;

    // println!("total and set bits: {} {}", total_bits, set_bits);

    // Iterate over all valid bitmasks
    while bitmask < (1 << total_bits) {
        // let cost = get_bitmask_cost(bitmask, bases, set_bits);
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
    // for i in 0..m {
    //     // let mut counts = HashMap::from([
    //     //     (b'A', 0), (b'T', 0), (b'C', 0), (b'G', 0), (b'*', 0)
    //     // ]);

    //     for j in 0..n {
    //         if j == 0 || (corr_mask & (1 << (j - 1))) != 0 {
    //             let base = bases[[j, i]];
    //             *counts.entry(base).or_insert(0) += 1;
    //         }
    //     }

    //     // Select the most frequent base
    //     if let Some((&best_base, _)) = counts.iter().max_by_key(|entry| entry.1) {
    //         corrections[i] = BASES_UPPER[best_base as usize];
    //     }
    // }

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

        // println!("counts: {:#?}", counts);
    
        // Select the most frequent base
        // if let Some((best_index, _)) = counts.iter().enumerate().max_by_key(|(_, &count)| count) {
        //     // println!("best idx {}", best_index);
        //     corrections[i] = BASES_UPPER_COUNTER2[best_index as usize];
        // }

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
            }
        }
        
        
    }
    
    corrections
}



/// Finds the optimal correction sequence using the best partition
fn naive_modified_mec_original(bases: &Array2<u8>) -> Vec<u8> {
    let n = bases.nrows();
    let m = bases.ncols();

    // println!("bases: {:#?}", bases);
    // println!("size of the matrix: {} {}", n, m);

    let total_bits = n as u32;
    let mut bitmask = 0;

    let mut min_cost = u32::MAX;
    let mut corr_mask = 0;

    // Iterate over all valid bitmasks
    while bitmask < (1 << total_bits) {
        let cost = get_bitmask_cost_original(bitmask, bases);
        if cost < min_cost {
            min_cost = cost;
            corr_mask = bitmask;
        }

        // Generate the next bitmask 
        bitmask += 1;
    }

    // Compute the corrected sequence
    let mut corrections: Vec<u8> = vec![4; m]; // Default to '*' if no match

    for i in 0..m {
        let mut counts: Vec<u8> = vec![0; 5]; // Assuming 5 possible bases (A, T, C, G, '*')
    
        for j in 0..n {
            if((corr_mask&1 == 0) && (corr_mask&(1<<j)) == 0) {
                let base = bases[[j, i]];
                if base != BASES_MAP[b'.' as usize] {
                    counts[BASES_UPPER_COUNTER[base as usize]] += 1;
                }
            }
            else if((corr_mask&1 != 0) && (corr_mask&(1<<j)) != 0) {
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
            }
        }
        
        
    }
    
    corrections
}


/// Finds the optimal correction sequence using the best partition
fn naive_modified_mec2(bases: &Array2<u8>) -> u32 {
    let n = bases.nrows();
    let m = bases.ncols();

    // println!("bases: {:#?}", bases);
    // println!("size of the matrix: {} {}", n, m);

    let total_bits = (n - 1) as u32;
    let set_bits = min(6, total_bits / 3); // max 6 rows, or n/3
    // let set_bits = max(2, total_bits+3 / 5); // max 6 rows, or n/3
    let mut bitmask = (1 << set_bits) - 1; // Smallest bitmask with `set_bits` bits set

    let mut min_cost = u32::MAX;
    let mut corr_mask = 0;

    // println!("total and set bits: {} {}", total_bits, set_bits);

    // Iterate over all valid bitmasks
    while bitmask < (1 << total_bits) {
        // let cost = get_bitmask_cost(bitmask, bases, set_bits);
        let cost = get_bitmask_cost(bitmask, bases, set_bits);
        if cost < min_cost {
            min_cost = cost;
            corr_mask = bitmask;
        }

        // Generate the next bitmask with the same number of set bits
        let t = bitmask | (bitmask - 1);
        bitmask = (t + 1) | (((!t & (!t).wrapping_neg()) - 1) >> (bitmask.trailing_zeros() + 1));
    }

    bitmask
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





/*#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use approx::assert_relative_eq;
    use ndarray::{Array1, Array3};

    use super::{inference, prepare_examples};

    #[test]
    fn test() {
        let _guard = tch::no_grad_guard();
        let device = tch::Device::Cpu;
        let resources = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        // Load model
        let mut model =
            tch::CModule::load_on_device(&resources.join("resources/mm2-attn.pt"), device).unwrap();
        model.set_eval();

        // Get files list
        /*let mut files: Vec<_> = resources
            .join("resources/example_feats")
            .read_dir()
            .unwrap()
            .filter_map(|p| {
                let p = p.unwrap().path();
                p.
                match p.extension() {
                    Some(ext) if ext == "npy" => Some(p),
                    _ => None,
                }
            })
            .collect();
        files.sort();*/

        // Create input data
        let features: Vec<_> = (0..4)
            .into_iter()
            .map(|wid| {
                let feats: Array3<u8> =
                    read_npy(format!("resources/example_feats/{}.features.npy", wid)).unwrap();
                let supported: Array1<u16> =
                    read_npy(format!("resources/example_feats/{}.supported.npy", wid)).unwrap();
                (feats, supported.iter().map(|s| *s as usize).collect())
            })
            .collect();
        let mut input_data = prepare_examples(0, features);
        let batch = input_data.batches.remove(0);

        let output = inference(batch, &model, device);
        let predicted: Array1<f32> = output
            .1
            .into_iter()
            .flat_map(|l| Vec::try_from(l).unwrap().into_iter())
            .collect();

        let target: Array1<f32> =
            read_npy(resources.join("resources/example_feats_tch_out.npy")).unwrap();

        assert_relative_eq!(predicted, target, epsilon = 1e-5);
    }

    #[test]
    fn test2() {
        let _guard = tch::no_grad_guard();
        let device = tch::Device::Cpu;
        let resources = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        // Load model
        let mut model =
            tch::CModule::load_on_device(&resources.join("resources/model.pt"), device).unwrap();
        model.set_eval();

        // Get files list
        let mut files: Vec<_> = PathBuf::from("resources/test_rs")
            .read_dir()
            .unwrap()
            .filter_map(|p| {
                let p = p.unwrap().path();
                match p.extension() {
                    Some(ext) if ext == "npy" => Some(p),
                    _ => None,
                }
            })
            .collect();
        files.sort();

        // Create input data
        let mut features: Vec<_> = files
            .into_iter()
            .enumerate()
            .map(|(i, p)| {
                let feats: Array3<u8> = read_npy(p).unwrap();
                (i as u16, feats)
            })
            .collect();
        let input_data = prepare_examples(0, &mut features);

        let output = inference(input_data, &model, device);
        let predicted: Array1<f32> = output
            .windows
            .into_iter()
            .flat_map(|(_, _, l)| l.into_iter())
            .collect();

        println!("{:?}", &predicted.to_vec()[4056 - 5..4056 + 5]);
    }
}*/
