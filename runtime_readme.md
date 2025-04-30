# Runtime (Reported for chr9 HG002 60X hifi data)


## All-vs-all overlap:
### Command Used
```bash 
herro/scripts/create_batched_alignments.sh chr9_HG002_HiFi_preprocess_herro.fastq.gz chr9_HG002_HiFi_preprocess_herro.read_ids 64 batch_alignments
```
### Runtime : 26 mins 19 seconds
- <h4> log file output: </h4>
```bash
[M::main] Version: 2.26-r1175
[M::main] CMD: minimap2 -K8g -cx ava-ont -k25 -w17 -e200 -r150 -m2500 -z200 -f 0.005 -t64 --dual=yes chr9_HG002_HiFi_preprocess_herro.fastq.gz chr9_HG002_HiFi_preprocess_herro.fastq.gz
[M::main] Real time: 1579.420 sec; CPU: 82996.115 sec; Peak RSS: 54.148 GB
49121110it [26:18, 31109.80it/s]
82807.07user 252.91system 26:20.76elapsed 5254%CPU (0avgtext+0avgdata 56777940maxresident)k
754562inputs+0outputs (64major+53646731minor)pagefaults 0swaps
```




## HALE
### Command Used
```bash
./hale inference --read-alns batch_alignments -m "hale" -t 64 chr9_HG002_HiFi_preprocess_herro.fastq.gz corrected_reads.fasta
```
### Runtime : 8 mins 52 seconds
- <h4> Using 64 threads but CPU utilization is only 4311%, offering a significant scope. </h4>
- <h4> log file output: </h4>
```bash
[00:00:44] Parsed 447493 reads.
[00:08:07] Processed 434815 reads.
22876.14user 69.53system 8:52.24elapsed 4311%CPU (0avgtext+0avgdata 50373996maxresident)k
6919346inputs+15223824outputs (375major+6575177minor)pagefaults 0swaps
```





## HERRO:
### Command Used
```bash
model="/pscratch/sd/p/pbarak/model_R10_v0.1.pt"
./herro inference --read-alns batch_alignments -t 8 -d 0,1,2,3 -m $model -b 64 chr9_preprocess_herro.fastq.gz herro_corrected_reads.fasta
```
### Runtime : 10 mins 24 seconds
- <h4> log file output: </h4>
```bash
[00:00:44] Parsed 447493 reads.  
[00:09:34] Processed 434815 reads.
11895.54user 79.59system 10:24.76elapsed 1922%CPU (0avgtext+0avgdata 25456944maxresident)k
9218976inputs+15223368outputs (24647major+8526811minor)pagefaults 0swaps
```




## Hifiasm

### Command Used
```bash
./hifiasm -t 64 -o hifiasm_op2/chr9_hifi.asm --write-paf --write-ec temp  chr9_HG002_HiFi_preprocess_herro.fastq.gz
```
### Runtime : 16 mins 45 seconds
- <h4> log file output: </h4>
```bash
[M::main] Version: 0.25.0-r726
[M::main] CMD: /pscratch/sd/p/pbarak/tools/hifiasm/./hifiasm -t 64 -o hifiasm_op2/chr9_hifi.asm --write-paf --write-ec temp chr9_HG002_HiFi_preprocess_herro.fastq.gz
[M::main] Real time: 1004.542 sec; CPU: 44218.559 sec; Peak RSS: 17.752 GB
```







<!-- 
Full logs:


########################################################################
#########################  ALL-vs-All:  ########################
########################################################################

/usr/bin/time /pscratch/sd/p/pbarak/tools/herro/scripts/create_batched_alignments.sh chr9_HG002_HiFi_preprocess_herro.fastq.gz chr9_HG002_HiFi_preprocess_herro.read_ids 64 batch_alignments2 
0it [00:00, ?it/s][M::mm_idx_gen::82.827*1.99] collected minimizers
[M::mm_idx_gen::89.277*3.87] sorted minimizers
[M::main::89.277*3.87] loaded/built the index for 445707 target sequence(s)
[M::mm_mapopt_update::89.872*3.85] mid_occ = 213
[M::mm_idx_stat] kmer size: 25; skip: 17; is_hpc: 0; #seq: 445707
[M::mm_idx_stat::90.246*3.84] distinct minimizers: 32976004 (55.46% are singletons); average occurrences: 27.077; average spacing: 8.961; total length: 8001495594
48747732it [25:04, 560369.37it/s][M::worker_pipeline::1506.444*54.48] mapped 445623 sequences
48909687it [25:05, 180690.68it/s][M::worker_pipeline::1507.921*54.43] mapped 1870 sequences
[M::mm_idx_gen::1508.833*54.40] collected minimizers
[M::mm_idx_gen::1508.857*54.40] sorted minimizers
[M::main::1508.857*54.40] loaded/built the index for 1786 target sequence(s)
[M::mm_mapopt_update::1508.857*54.40] mid_occ = 213
[M::mm_idx_stat] kmer size: 25; skip: 17; is_hpc: 0; #seq: 1786
[M::mm_idx_stat::1508.868*54.40] distinct minimizers: 688986 (17.92% are singletons); average occurrences: 5.171; average spacing: 8.989; total length: 32025800
49097157it [26:16, 11314.94it/s][M::worker_pipeline::1578.149*52.59] mapped 445623 sequences
[M::worker_pipeline::1578.336*52.58] mapped 1870 sequences
[M::main] Version: 2.26-r1175
[M::main] CMD: minimap2 -K8g -cx ava-ont -k25 -w17 -e200 -r150 -m2500 -z200 -f 0.005 -t64 --dual=yes chr9_HG002_HiFi_preprocess_herro.fastq.gz chr9_HG002_HiFi_preprocess_herro.fastq.gz
[M::main] Real time: 1579.420 sec; CPU: 82996.115 sec; Peak RSS: 54.148 GB
49121110it [26:18, 31109.80it/s]
82807.07user 252.91system 26:20.76elapsed 5254%CPU (0avgtext+0avgdata 56777940maxresident)k
754562inputs+0outputs (64major+53646731minor)pagefaults 0swaps





########################################################################
#########################  Hifiasm:  ########################
########################################################################

/pscratch/sd/p/pbarak/tools/hifiasm/./hifiasm -t 64 -o hifiasm_op2/chr9_hifi.asm --write-paf --write-ec temp  "chr9_HG002_HiFi_preprocess_herro.fastq.gz"
[M::ha_analyze_count] lowest: count[5] = 33059
[M::ha_analyze_count] highest: count[63] = 2927284
[M::ha_hist_line]     2: ********************* 615568
[M::ha_hist_line]     3: **** 103556
[M::ha_hist_line]     4: ** 49769
[M::ha_hist_line]     5: * 33059
[M::ha_hist_line]     6: * 35549
[M::ha_hist_line]     7: ** 46582
[M::ha_hist_line]     8: ** 60921
[M::ha_hist_line]     9: *** 88412
[M::ha_hist_line]    10: **** 113952
[M::ha_hist_line]    11: ***** 138446
[M::ha_hist_line]    12: ***** 154673
[M::ha_hist_line]    13: ****** 162102
[M::ha_hist_line]    14: ****** 166217
[M::ha_hist_line]    15: ****** 162012
[M::ha_hist_line]    16: ***** 153036
[M::ha_hist_line]    17: ***** 146687
[M::ha_hist_line]    18: ***** 155610
[M::ha_hist_line]    19: ****** 174295
[M::ha_hist_line]    20: ******* 194804
[M::ha_hist_line]    21: ******** 228159
[M::ha_hist_line]    22: ********* 275996
[M::ha_hist_line]    23: *********** 328438
[M::ha_hist_line]    24: ************* 383765
[M::ha_hist_line]    25: *************** 437300
[M::ha_hist_line]    26: ***************** 494389
[M::ha_hist_line]    27: ******************* 551439
[M::ha_hist_line]    28: ******************** 589001
[M::ha_hist_line]    29: ********************* 628236
[M::ha_hist_line]    30: ********************** 648649
[M::ha_hist_line]    31: ********************** 656199
[M::ha_hist_line]    32: ********************** 643543
[M::ha_hist_line]    33: ********************* 629362
[M::ha_hist_line]    34: ********************* 624175
[M::ha_hist_line]    35: ******************** 599524
[M::ha_hist_line]    36: ******************** 577418
[M::ha_hist_line]    37: ******************** 571428
[M::ha_hist_line]    38: ******************* 562565
[M::ha_hist_line]    39: ******************* 549365
[M::ha_hist_line]    40: ****************** 539264
[M::ha_hist_line]    41: ****************** 531108
[M::ha_hist_line]    42: ******************* 558502
[M::ha_hist_line]    43: ********************* 605986
[M::ha_hist_line]    44: *********************** 686287
[M::ha_hist_line]    45: ************************** 752969
[M::ha_hist_line]    46: ***************************** 858054
[M::ha_hist_line]    47: ********************************* 964208
[M::ha_hist_line]    48: ************************************* 1077628
[M::ha_hist_line]    49: ***************************************** 1192148
[M::ha_hist_line]    50: ********************************************** 1332977
[M::ha_hist_line]    51: *************************************************** 1494282
[M::ha_hist_line]    52: ******************************************************** 1649481
[M::ha_hist_line]    53: *************************************************************** 1830271
[M::ha_hist_line]    54: ******************************************************************* 1964601
[M::ha_hist_line]    55: ************************************************************************ 2121057
[M::ha_hist_line]    56: ****************************************************************************** 2290878
[M::ha_hist_line]    57: ************************************************************************************ 2464309
[M::ha_hist_line]    58: **************************************************************************************** 2572955
[M::ha_hist_line]    59: ******************************************************************************************** 2687382
[M::ha_hist_line]    60: *********************************************************************************************** 2772028
[M::ha_hist_line]    61: ************************************************************************************************** 2858044
[M::ha_hist_line]    62: *************************************************************************************************** 2906513
[M::ha_hist_line]    63: **************************************************************************************************** 2927284
[M::ha_hist_line]    64: *************************************************************************************************** 2893077
[M::ha_hist_line]    65: ************************************************************************************************ 2823991
[M::ha_hist_line]    66: ********************************************************************************************* 2729869
[M::ha_hist_line]    67: ***************************************************************************************** 2615211
[M::ha_hist_line]    68: ************************************************************************************ 2461355
[M::ha_hist_line]    69: ****************************************************************************** 2288510
[M::ha_hist_line]    70: *********************************************************************** 2090133
[M::ha_hist_line]    71: **************************************************************** 1885815
[M::ha_hist_line]    72: ************************************************************ 1753765
[M::ha_hist_line]    73: ******************************************************** 1626030
[M::ha_hist_line]    74: ************************************************** 1458666
[M::ha_hist_line]    75: ******************************************* 1269209
[M::ha_hist_line]    76: ************************************* 1096396
[M::ha_hist_line]    77: ******************************** 935878
[M::ha_hist_line]    78: *************************** 802852
[M::ha_hist_line]    79: *********************** 676705
[M::ha_hist_line]    80: ******************* 563622
[M::ha_hist_line]    81: **************** 468597
[M::ha_hist_line]    82: ************* 384456
[M::ha_hist_line]    83: *********** 311451
[M::ha_hist_line]    84: ******** 247431
[M::ha_hist_line]    85: ******* 200596
[M::ha_hist_line]    86: ***** 160434
[M::ha_hist_line]    87: **** 127614
[M::ha_hist_line]    88: *** 100776
[M::ha_hist_line]    89: *** 81426
[M::ha_hist_line]    90: ** 68182
[M::ha_hist_line]    91: ** 58797
[M::ha_hist_line]    92: ** 46457
[M::ha_hist_line]    93: * 40822
[M::ha_hist_line]    94: * 31882
[M::ha_hist_line]    95: * 28847
[M::ha_hist_line]    96: * 28487
[M::ha_hist_line]    97: * 27356
[M::ha_hist_line]    98: * 24481
[M::ha_hist_line]    99: * 23499
[M::ha_hist_line]   100: * 24136
[M::ha_hist_line]   101: * 22939
[M::ha_hist_line]   102: * 24048
[M::ha_hist_line]   103: * 21746
[M::ha_hist_line]   104: * 23796
[M::ha_hist_line]   105: * 21143
[M::ha_hist_line]   106: * 21867
[M::ha_hist_line]   107: * 20890
[M::ha_hist_line]   108: * 20858
[M::ha_hist_line]   109: * 20745
[M::ha_hist_line]   110: * 23253
[M::ha_hist_line]   111: * 22570
[M::ha_hist_line]   112: * 22356
[M::ha_hist_line]   113: * 24778
[M::ha_hist_line]   114: * 26497
[M::ha_hist_line]   115: * 25972
[M::ha_hist_line]   116: * 25306
[M::ha_hist_line]   117: * 27473
[M::ha_hist_line]   118: * 26261
[M::ha_hist_line]   119: * 28308
[M::ha_hist_line]   120: * 30457
[M::ha_hist_line]   121: * 31198
[M::ha_hist_line]   122: * 34005
[M::ha_hist_line]   123: * 33075
[M::ha_hist_line]   124: * 34270
[M::ha_hist_line]   125: * 33817
[M::ha_hist_line]   126: * 33368
[M::ha_hist_line]   127: * 33083
[M::ha_hist_line]   128: * 32859
[M::ha_hist_line]   129: * 31387
[M::ha_hist_line]   130: * 32421
[M::ha_hist_line]   131: * 31016
[M::ha_hist_line]   132: * 29821
[M::ha_hist_line]   133: * 28675
[M::ha_hist_line]   134: * 29121
[M::ha_hist_line]   135: * 27377
[M::ha_hist_line]   136: * 26840
[M::ha_hist_line]   137: * 24457
[M::ha_hist_line]   138: * 23873
[M::ha_hist_line]   139: * 23379
[M::ha_hist_line]   140: * 22575
[M::ha_hist_line]   141: * 19454
[M::ha_hist_line]   142: * 17745
[M::ha_hist_line]   143: * 18297
[M::ha_hist_line]   144: * 16735
[M::ha_hist_line]   145: * 16123
[M::ha_hist_line]   146: * 14878
[M::ha_hist_line]  rest: ************************************************ 1399335
[M::ha_analyze_count] left: count[31] = 656199
[M::ha_analyze_count] right: none
[M::ha_ft_gen] peak_hom: 63; peak_het: 31
[M::ha_ct_shrink::113.825*5.05] ==> counted 232311 distinct minimizer k-mers
[M::ha_ft_gen::114.036*5.04@17.752GB] ==> filtered out 232311 k-mers occurring 315 or more times
[M::ha_opt_update_cov] updated max_n_chain to 315
[M::yak_count] collected 0 minimizers
[M::yak_count] collected 217927726 minimizers
[M::ha_pt_gen::174.238*5.66] ==> counted 5969623 distinct minimizer k-mers
[M::ha_pt_gen] count[4095] = 0 (for sanity check)
[M::ha_analyze_count] lowest: count[5] = 3311
[M::ha_analyze_count] highest: count[63] = 114825
[M::ha_hist_line]     1: ****************************************************************************************************> 2272136
[M::ha_hist_line]     2: ************************************************ 55676
[M::ha_hist_line]     3: ********* 10433
[M::ha_hist_line]     4: **** 4787
[M::ha_hist_line]     5: *** 3311
[M::ha_hist_line]     6: *** 3548
[M::ha_hist_line]     7: **** 4608
[M::ha_hist_line]     8: ***** 5820
[M::ha_hist_line]     9: ******* 8014
[M::ha_hist_line]    10: ********* 10247
[M::ha_hist_line]    11: ********** 11935
[M::ha_hist_line]    12: ************ 13255
[M::ha_hist_line]    13: ************ 13741
[M::ha_hist_line]    14: ************ 13841
[M::ha_hist_line]    15: *********** 12456
[M::ha_hist_line]    16: ********** 11431
[M::ha_hist_line]    17: ********* 10467
[M::ha_hist_line]    18: ********* 10454
[M::ha_hist_line]    19: ********* 10872
[M::ha_hist_line]    20: ********** 11462
[M::ha_hist_line]    21: *********** 12818
[M::ha_hist_line]    22: ************* 14822
[M::ha_hist_line]    23: *************** 17204
[M::ha_hist_line]    24: ****************** 20197
[M::ha_hist_line]    25: ******************** 22618
[M::ha_hist_line]    26: ********************** 25347
[M::ha_hist_line]    27: ************************ 28002
[M::ha_hist_line]    28: ************************** 29613
[M::ha_hist_line]    29: *************************** 30978
[M::ha_hist_line]    30: **************************** 32066
[M::ha_hist_line]    31: **************************** 32085
[M::ha_hist_line]    32: *************************** 31305
[M::ha_hist_line]    33: ************************** 30096
[M::ha_hist_line]    34: ************************** 29712
[M::ha_hist_line]    35: ************************* 28363
[M::ha_hist_line]    36: *********************** 26962
[M::ha_hist_line]    37: *********************** 26602
[M::ha_hist_line]    38: ********************** 25548
[M::ha_hist_line]    39: ********************** 24696
[M::ha_hist_line]    40: ********************* 23897
[M::ha_hist_line]    41: ******************** 23240
[M::ha_hist_line]    42: ********************* 24279
[M::ha_hist_line]    43: *********************** 25951
[M::ha_hist_line]    44: ************************* 29021
[M::ha_hist_line]    45: *************************** 31440
[M::ha_hist_line]    46: ******************************* 35398
[M::ha_hist_line]    47: *********************************** 39646
[M::ha_hist_line]    48: ************************************** 43809
[M::ha_hist_line]    49: ****************************************** 48414
[M::ha_hist_line]    50: *********************************************** 54033
[M::ha_hist_line]    51: ***************************************************** 60425
[M::ha_hist_line]    52: ********************************************************** 66245
[M::ha_hist_line]    53: **************************************************************** 73556
[M::ha_hist_line]    54: ********************************************************************* 78759
[M::ha_hist_line]    55: ************************************************************************** 84763
[M::ha_hist_line]    56: ******************************************************************************** 91984
[M::ha_hist_line]    57: ************************************************************************************* 97807
[M::ha_hist_line]    58: ***************************************************************************************** 102303
[M::ha_hist_line]    59: ******************************************************************************************** 106213
[M::ha_hist_line]    60: *********************************************************************************************** 109611
[M::ha_hist_line]    61: ************************************************************************************************** 112222
[M::ha_hist_line]    62: *************************************************************************************************** 113760
[M::ha_hist_line]    63: **************************************************************************************************** 114825
[M::ha_hist_line]    64: ************************************************************************************************** 112829
[M::ha_hist_line]    65: ************************************************************************************************ 110331
[M::ha_hist_line]    66: ********************************************************************************************* 106362
[M::ha_hist_line]    67: **************************************************************************************** 100867
[M::ha_hist_line]    68: *********************************************************************************** 95465
[M::ha_hist_line]    69: ***************************************************************************** 88349
[M::ha_hist_line]    70: ********************************************************************** 80529
[M::ha_hist_line]    71: **************************************************************** 73317
[M::ha_hist_line]    72: *********************************************************** 67642
[M::ha_hist_line]    73: ******************************************************* 62868
[M::ha_hist_line]    74: ************************************************* 56232
[M::ha_hist_line]    75: ****************************************** 48619
[M::ha_hist_line]    76: ************************************ 41881
[M::ha_hist_line]    77: ******************************* 35631
[M::ha_hist_line]    78: *************************** 30745
[M::ha_hist_line]    79: *********************** 25872
[M::ha_hist_line]    80: ******************* 21697
[M::ha_hist_line]    81: **************** 18079
[M::ha_hist_line]    82: ************* 14783
[M::ha_hist_line]    83: *********** 12104
[M::ha_hist_line]    84: ********* 9838
[M::ha_hist_line]    85: ******* 7930
[M::ha_hist_line]    86: ****** 6433
[M::ha_hist_line]    87: ***** 5271
[M::ha_hist_line]    88: **** 4335
[M::ha_hist_line]    89: *** 3600
[M::ha_hist_line]    90: *** 3141
[M::ha_hist_line]    91: ** 2823
[M::ha_hist_line]    92: ** 2347
[M::ha_hist_line]    93: ** 2062
[M::ha_hist_line]    94: ** 1764
[M::ha_hist_line]    95: * 1680
[M::ha_hist_line]    96: * 1722
[M::ha_hist_line]    97: * 1718
[M::ha_hist_line]    98: * 1464
[M::ha_hist_line]    99: * 1530
[M::ha_hist_line]   100: * 1479
[M::ha_hist_line]   101: * 1522
[M::ha_hist_line]   102: * 1486
[M::ha_hist_line]   103: * 1341
[M::ha_hist_line]   104: * 1388
[M::ha_hist_line]   105: * 1372
[M::ha_hist_line]   106: * 1410
[M::ha_hist_line]   107: * 1235
[M::ha_hist_line]   108: * 1260
[M::ha_hist_line]   109: * 1268
[M::ha_hist_line]   110: * 1381
[M::ha_hist_line]   111: * 1249
[M::ha_hist_line]   112: * 1306
[M::ha_hist_line]   113: * 1424
[M::ha_hist_line]   114: * 1482
[M::ha_hist_line]   115: * 1491
[M::ha_hist_line]   116: * 1311
[M::ha_hist_line]   117: * 1483
[M::ha_hist_line]   118: * 1394
[M::ha_hist_line]   119: * 1491
[M::ha_hist_line]   120: * 1626
[M::ha_hist_line]   121: * 1588
[M::ha_hist_line]   122: * 1710
[M::ha_hist_line]   123: * 1723
[M::ha_hist_line]   124: * 1712
[M::ha_hist_line]   125: ** 1727
[M::ha_hist_line]   126: * 1648
[M::ha_hist_line]   127: * 1673
[M::ha_hist_line]   128: * 1686
[M::ha_hist_line]   129: * 1526
[M::ha_hist_line]   130: * 1629
[M::ha_hist_line]   131: * 1580
[M::ha_hist_line]   132: * 1465
[M::ha_hist_line]   133: * 1423
[M::ha_hist_line]   134: * 1478
[M::ha_hist_line]   135: * 1437
[M::ha_hist_line]   136: * 1272
[M::ha_hist_line]   137: * 1222
[M::ha_hist_line]   138: * 1282
[M::ha_hist_line]   139: * 1160
[M::ha_hist_line]   140: * 1101
[M::ha_hist_line]   141: * 1044
[M::ha_hist_line]   142: * 962
[M::ha_hist_line]   143: * 1020
[M::ha_hist_line]   144: * 900
[M::ha_hist_line]   145: * 881
[M::ha_hist_line]   146: * 801
[M::ha_hist_line]   147: * 830
[M::ha_hist_line]   148: * 757
[M::ha_hist_line]   149: * 770
[M::ha_hist_line]   150: * 749
[M::ha_hist_line]   151: * 749
[M::ha_hist_line]   152: * 763
[M::ha_hist_line]   153: * 789
[M::ha_hist_line]   154: * 830
[M::ha_hist_line]   155: * 864
[M::ha_hist_line]   156: * 739
[M::ha_hist_line]   157: * 714
[M::ha_hist_line]   158: * 720
[M::ha_hist_line]   159: * 674
[M::ha_hist_line]   160: * 756
[M::ha_hist_line]   161: * 743
[M::ha_hist_line]   162: * 676
[M::ha_hist_line]   163: * 615
[M::ha_hist_line]   164: * 637
[M::ha_hist_line]   165: * 594
[M::ha_hist_line]   166: * 657
[M::ha_hist_line]   167: * 592
[M::ha_hist_line]  rest: ****************************************** 48367
[M::ha_analyze_count] left: count[31] = 32085
[M::ha_analyze_count] right: none
[M::ha_pt_gen] peak_hom: 63; peak_het: 31
[M::ha_ct_shrink::174.260*5.66] ==> counted 3697487 distinct minimizer k-mers
[M::ha_pt_gen::] counting in normal mode
[M::yak_count] collected 217927726 minimizers
[M::yak_count] collected 0 minimizers
[M::ha_pt_gen::187.601*7.28] ==> indexed 215655590 positions, counted 3697487 distinct minimizer k-mers
[M::pec::237.768] # bases: 8033521394; # corrected bases: 12179212
[M::pec::1.272] # exact o: 29463324; # non-exact o: 2749905
[M::ha_assemble::427.227*39.06@17.752GB] ==> corrected reads for round 1
[M::ha_assemble] # bases: 8033521394; # corrected bases: 12179212
[M::yak_count] collected 217857696 minimizers
[M::yak_count] collected 0 minimizers
[M::ha_pt_gen::441.801*38.58] ==> counted 3863184 distinct minimizer k-mers
[M::ha_pt_gen] count[4095] = 0 (for sanity check)
[M::ha_analyze_count] lowest: count[5] = 1499
[M::ha_analyze_count] highest: count[63] = 114670
[M::ha_hist_line]     1: ****************************************************************************************************> 236583
[M::ha_hist_line]     2: *** 3153
[M::ha_hist_line]     3: * 1582
[M::ha_hist_line]     4: * 1496
[M::ha_hist_line]     5: * 1499
[M::ha_hist_line]     6: ** 2435
[M::ha_hist_line]     7: *** 3473
[M::ha_hist_line]     8: **** 5119
[M::ha_hist_line]     9: ****** 7167
[M::ha_hist_line]    10: ******** 9643
[M::ha_hist_line]    11: ********** 11395
[M::ha_hist_line]    12: *********** 12810
[M::ha_hist_line]    13: ************ 13638
[M::ha_hist_line]    14: ************ 13554
[M::ha_hist_line]    15: *********** 12835
[M::ha_hist_line]    16: ********** 11240
[M::ha_hist_line]    17: ********* 10004
[M::ha_hist_line]    18: ********* 10147
[M::ha_hist_line]    19: ********* 10674
[M::ha_hist_line]    20: ********* 10852
[M::ha_hist_line]    21: *********** 12147
[M::ha_hist_line]    22: ************ 13860
[M::ha_hist_line]    23: ************** 16207
[M::ha_hist_line]    24: ***************** 19195
[M::ha_hist_line]    25: ******************* 21468
[M::ha_hist_line]    26: ********************* 24410
[M::ha_hist_line]    27: ************************ 27205
[M::ha_hist_line]    28: ************************* 28908
[M::ha_hist_line]    29: *************************** 30612
[M::ha_hist_line]    30: **************************** 31997
[M::ha_hist_line]    31: **************************** 31703
[M::ha_hist_line]    32: *************************** 31191
[M::ha_hist_line]    33: ************************** 30177
[M::ha_hist_line]    34: ************************** 30015
[M::ha_hist_line]    35: ************************* 28378
[M::ha_hist_line]    36: ************************ 27190
[M::ha_hist_line]    37: *********************** 26403
[M::ha_hist_line]    38: *********************** 25839
[M::ha_hist_line]    39: ********************* 24615
[M::ha_hist_line]    40: ********************* 23862
[M::ha_hist_line]    41: ******************** 23019
[M::ha_hist_line]    42: ******************** 23166
[M::ha_hist_line]    43: ********************** 24896
[M::ha_hist_line]    44: ************************ 26970
[M::ha_hist_line]    45: ************************** 29673
[M::ha_hist_line]    46: ***************************** 33162
[M::ha_hist_line]    47: ******************************** 37251
[M::ha_hist_line]    48: ************************************ 40994
[M::ha_hist_line]    49: **************************************** 45390
[M::ha_hist_line]    50: ******************************************** 50694
[M::ha_hist_line]    51: ************************************************** 57047
[M::ha_hist_line]    52: ****************************************************** 62461
[M::ha_hist_line]    53: ************************************************************* 69554
[M::ha_hist_line]    54: ***************************************************************** 74790
[M::ha_hist_line]    55: ********************************************************************** 80355
[M::ha_hist_line]    56: **************************************************************************** 86867
[M::ha_hist_line]    57: ********************************************************************************** 94161
[M::ha_hist_line]    58: ************************************************************************************** 98248
[M::ha_hist_line]    59: ******************************************************************************************* 103961
[M::ha_hist_line]    60: ********************************************************************************************* 107039
[M::ha_hist_line]    61: ************************************************************************************************ 110488
[M::ha_hist_line]    62: ************************************************************************************************** 112750
[M::ha_hist_line]    63: **************************************************************************************************** 114670
[M::ha_hist_line]    64: ************************************************************************************************** 112928
[M::ha_hist_line]    65: ************************************************************************************************* 111391
[M::ha_hist_line]    66: *********************************************************************************************** 108625
[M::ha_hist_line]    67: ******************************************************************************************* 104160
[M::ha_hist_line]    68: ************************************************************************************** 98639
[M::ha_hist_line]    69: ********************************************************************************* 92503
[M::ha_hist_line]    70: ************************************************************************** 84927
[M::ha_hist_line]    71: ******************************************************************* 76594
[M::ha_hist_line]    72: ************************************************************** 71580
[M::ha_hist_line]    73: ********************************************************** 66305
[M::ha_hist_line]    74: ***************************************************** 60315
[M::ha_hist_line]    75: ********************************************** 53302
[M::ha_hist_line]    76: **************************************** 46371
[M::ha_hist_line]    77: *********************************** 39829
[M::ha_hist_line]    78: ****************************** 34276
[M::ha_hist_line]    79: ************************* 28773
[M::ha_hist_line]    80: ********************* 24030
[M::ha_hist_line]    81: ****************** 20565
[M::ha_hist_line]    82: *************** 17158
[M::ha_hist_line]    83: ************ 13622
[M::ha_hist_line]    84: ********** 11289
[M::ha_hist_line]    85: ******** 9180
[M::ha_hist_line]    86: ******* 7481
[M::ha_hist_line]    87: ***** 6162
[M::ha_hist_line]    88: **** 4871
[M::ha_hist_line]    89: **** 4101
[M::ha_hist_line]    90: *** 3495
[M::ha_hist_line]    91: *** 3101
[M::ha_hist_line]    92: ** 2698
[M::ha_hist_line]    93: ** 2359
[M::ha_hist_line]    94: ** 1917
[M::ha_hist_line]    95: * 1668
[M::ha_hist_line]    96: * 1630
[M::ha_hist_line]    97: ** 1799
[M::ha_hist_line]    98: * 1591
[M::ha_hist_line]    99: * 1481
[M::ha_hist_line]   100: * 1471
[M::ha_hist_line]   101: * 1478
[M::ha_hist_line]   102: * 1527
[M::ha_hist_line]   103: * 1336
[M::ha_hist_line]   104: * 1418
[M::ha_hist_line]   105: * 1336
[M::ha_hist_line]   106: * 1365
[M::ha_hist_line]   107: * 1316
[M::ha_hist_line]   108: * 1323
[M::ha_hist_line]   109: * 1182
[M::ha_hist_line]   110: * 1392
[M::ha_hist_line]   111: * 1255
[M::ha_hist_line]   112: * 1222
[M::ha_hist_line]   113: * 1316
[M::ha_hist_line]   114: * 1394
[M::ha_hist_line]   115: * 1480
[M::ha_hist_line]   116: * 1374
[M::ha_hist_line]   117: * 1365
[M::ha_hist_line]   118: * 1465
[M::ha_hist_line]   119: * 1373
[M::ha_hist_line]   120: * 1439
[M::ha_hist_line]   121: * 1542
[M::ha_hist_line]   122: * 1613
[M::ha_hist_line]   123: * 1680
[M::ha_hist_line]   124: * 1703
[M::ha_hist_line]   125: * 1656
[M::ha_hist_line]   126: ** 1756
[M::ha_hist_line]   127: * 1691
[M::ha_hist_line]   128: * 1675
[M::ha_hist_line]   129: * 1599
[M::ha_hist_line]   130: * 1690
[M::ha_hist_line]   131: * 1564
[M::ha_hist_line]   132: * 1500
[M::ha_hist_line]   133: * 1520
[M::ha_hist_line]   134: * 1581
[M::ha_hist_line]   135: * 1463
[M::ha_hist_line]   136: * 1370
[M::ha_hist_line]   137: * 1266
[M::ha_hist_line]   138: * 1232
[M::ha_hist_line]   139: * 1148
[M::ha_hist_line]   140: * 1169
[M::ha_hist_line]   141: * 1189
[M::ha_hist_line]   142: * 1076
[M::ha_hist_line]   143: * 1082
[M::ha_hist_line]   144: * 956
[M::ha_hist_line]   145: * 884
[M::ha_hist_line]   146: * 920
[M::ha_hist_line]   147: * 817
[M::ha_hist_line]   148: * 811
[M::ha_hist_line]   149: * 710
[M::ha_hist_line]   150: * 784
[M::ha_hist_line]   151: * 743
[M::ha_hist_line]   152: * 695
[M::ha_hist_line]   153: * 801
[M::ha_hist_line]   154: * 749
[M::ha_hist_line]   155: * 796
[M::ha_hist_line]   156: * 793
[M::ha_hist_line]   157: * 762
[M::ha_hist_line]   158: * 816
[M::ha_hist_line]   159: * 705
[M::ha_hist_line]   160: * 679
[M::ha_hist_line]   161: * 721
[M::ha_hist_line]   162: * 783
[M::ha_hist_line]   163: * 737
[M::ha_hist_line]   164: * 626
[M::ha_hist_line]   165: * 586
[M::ha_hist_line]   166: * 632
[M::ha_hist_line]   167: * 629
[M::ha_hist_line]   168: * 592
[M::ha_hist_line]   169: * 582
[M::ha_hist_line]   170: * 575
[M::ha_hist_line]   171: * 612
[M::ha_hist_line]  rest: ***************************************** 47093
[M::ha_analyze_count] left: count[30] = 31997
[M::ha_analyze_count] right: none
[M::ha_pt_gen] peak_hom: 63; peak_het: 30
[M::ha_ct_shrink::441.829*38.57] ==> counted 3626601 distinct minimizer k-mers
[M::ha_pt_gen::] counting in normal mode
[M::yak_count] collected 217857696 minimizers
[M::yak_count] collected 0 minimizers
[M::ha_pt_gen::454.254*38.34] ==> indexed 217621113 positions, counted 3626601 distinct minimizer k-mers
[M::pec::176.501] # bases: 8034997463; # corrected bases: 415080
[M::pec::1.455] # exact o: 32189953; # non-exact o: 1099111
[M::ha_assemble::632.520*45.56@17.752GB] ==> corrected reads for round 2
[M::ha_assemble] # bases: 8034997463; # corrected bases: 415080
[M::yak_count] collected 217854112 minimizers
[M::yak_count] collected 0 minimizers
[M::ha_pt_gen::646.590*45.13] ==> counted 3788699 distinct minimizer k-mers
[M::ha_pt_gen] count[4095] = 0 (for sanity check)
[M::ha_analyze_count] lowest: count[5] = 1441
[M::ha_analyze_count] highest: count[63] = 114697
[M::ha_hist_line]     1: ****************************************************************************************************> 163468
[M::ha_hist_line]     2: ** 2430
[M::ha_hist_line]     3: * 1407
[M::ha_hist_line]     4: * 1411
[M::ha_hist_line]     5: * 1441
[M::ha_hist_line]     6: ** 2386
[M::ha_hist_line]     7: *** 3436
[M::ha_hist_line]     8: **** 4995
[M::ha_hist_line]     9: ****** 7219
[M::ha_hist_line]    10: ******** 9618
[M::ha_hist_line]    11: ********** 11406
[M::ha_hist_line]    12: *********** 12799
[M::ha_hist_line]    13: ************ 13540
[M::ha_hist_line]    14: ************ 13519
[M::ha_hist_line]    15: *********** 12837
[M::ha_hist_line]    16: ********** 11292
[M::ha_hist_line]    17: ********* 9972
[M::ha_hist_line]    18: ********* 10180
[M::ha_hist_line]    19: ********* 10623
[M::ha_hist_line]    20: ********* 10864
[M::ha_hist_line]    21: *********** 12122
[M::ha_hist_line]    22: ************ 13871
[M::ha_hist_line]    23: ************** 16105
[M::ha_hist_line]    24: ***************** 19134
[M::ha_hist_line]    25: ******************* 21395
[M::ha_hist_line]    26: ********************* 24269
[M::ha_hist_line]    27: ************************ 27244
[M::ha_hist_line]    28: ************************* 28842
[M::ha_hist_line]    29: *************************** 30566
[M::ha_hist_line]    30: **************************** 31980
[M::ha_hist_line]    31: **************************** 31790
[M::ha_hist_line]    32: *************************** 31028
[M::ha_hist_line]    33: ************************** 30183
[M::ha_hist_line]    34: ************************** 30045
[M::ha_hist_line]    35: ************************* 28309
[M::ha_hist_line]    36: ************************ 27280
[M::ha_hist_line]    37: *********************** 26396
[M::ha_hist_line]    38: *********************** 25899
[M::ha_hist_line]    39: ********************* 24576
[M::ha_hist_line]    40: ********************* 23821
[M::ha_hist_line]    41: ******************** 22981
[M::ha_hist_line]    42: ******************** 23145
[M::ha_hist_line]    43: ********************** 24909
[M::ha_hist_line]    44: *********************** 26908
[M::ha_hist_line]    45: ************************** 29604
[M::ha_hist_line]    46: ***************************** 33171
[M::ha_hist_line]    47: ******************************** 37128
[M::ha_hist_line]    48: ************************************ 40904
[M::ha_hist_line]    49: **************************************** 45348
[M::ha_hist_line]    50: ******************************************** 50632
[M::ha_hist_line]    51: ************************************************** 56897
[M::ha_hist_line]    52: ****************************************************** 62342
[M::ha_hist_line]    53: ************************************************************* 69435
[M::ha_hist_line]    54: ***************************************************************** 74682
[M::ha_hist_line]    55: ********************************************************************** 80180
[M::ha_hist_line]    56: **************************************************************************** 86693
[M::ha_hist_line]    57: ********************************************************************************** 93981
[M::ha_hist_line]    58: ************************************************************************************** 98221
[M::ha_hist_line]    59: ******************************************************************************************* 103939
[M::ha_hist_line]    60: ********************************************************************************************* 107045
[M::ha_hist_line]    61: ************************************************************************************************ 110489
[M::ha_hist_line]    62: ************************************************************************************************** 112759
[M::ha_hist_line]    63: **************************************************************************************************** 114697
[M::ha_hist_line]    64: ************************************************************************************************** 112931
[M::ha_hist_line]    65: ************************************************************************************************* 111415
[M::ha_hist_line]    66: *********************************************************************************************** 108673
[M::ha_hist_line]    67: ******************************************************************************************* 103982
[M::ha_hist_line]    68: ************************************************************************************** 98816
[M::ha_hist_line]    69: ********************************************************************************* 92591
[M::ha_hist_line]    70: ************************************************************************** 85104
[M::ha_hist_line]    71: ******************************************************************* 76573
[M::ha_hist_line]    72: *************************************************************** 71705
[M::ha_hist_line]    73: ********************************************************** 66528
[M::ha_hist_line]    74: ***************************************************** 60345
[M::ha_hist_line]    75: *********************************************** 53417
[M::ha_hist_line]    76: ***************************************** 46530
[M::ha_hist_line]    77: *********************************** 40015
[M::ha_hist_line]    78: ****************************** 34455
[M::ha_hist_line]    79: ************************* 28899
[M::ha_hist_line]    80: ********************* 24107
[M::ha_hist_line]    81: ****************** 20662
[M::ha_hist_line]    82: *************** 17242
[M::ha_hist_line]    83: ************ 13663
[M::ha_hist_line]    84: ********** 11351
[M::ha_hist_line]    85: ******** 9164
[M::ha_hist_line]    86: ******* 7519
[M::ha_hist_line]    87: ***** 6149
[M::ha_hist_line]    88: **** 4920
[M::ha_hist_line]    89: **** 4095
[M::ha_hist_line]    90: *** 3595
[M::ha_hist_line]    91: *** 3060
[M::ha_hist_line]    92: ** 2715
[M::ha_hist_line]    93: ** 2413
[M::ha_hist_line]    94: ** 1896
[M::ha_hist_line]    95: * 1694
[M::ha_hist_line]    96: * 1641
[M::ha_hist_line]    97: ** 1789
[M::ha_hist_line]    98: * 1614
[M::ha_hist_line]    99: * 1489
[M::ha_hist_line]   100: * 1471
[M::ha_hist_line]   101: * 1453
[M::ha_hist_line]   102: * 1572
[M::ha_hist_line]   103: * 1329
[M::ha_hist_line]   104: * 1408
[M::ha_hist_line]   105: * 1369
[M::ha_hist_line]   106: * 1349
[M::ha_hist_line]   107: * 1349
[M::ha_hist_line]   108: * 1306
[M::ha_hist_line]   109: * 1159
[M::ha_hist_line]   110: * 1405
[M::ha_hist_line]   111: * 1239
[M::ha_hist_line]   112: * 1201
[M::ha_hist_line]   113: * 1339
[M::ha_hist_line]   114: * 1439
[M::ha_hist_line]   115: * 1450
[M::ha_hist_line]   116: * 1383
[M::ha_hist_line]   117: * 1361
[M::ha_hist_line]   118: * 1425
[M::ha_hist_line]   119: * 1378
[M::ha_hist_line]   120: * 1465
[M::ha_hist_line]   121: * 1537
[M::ha_hist_line]   122: * 1633
[M::ha_hist_line]   123: * 1657
[M::ha_hist_line]   124: * 1694
[M::ha_hist_line]   125: * 1646
[M::ha_hist_line]   126: ** 1761
[M::ha_hist_line]   127: * 1661
[M::ha_hist_line]   128: * 1671
[M::ha_hist_line]   129: * 1597
[M::ha_hist_line]   130: * 1697
[M::ha_hist_line]   131: * 1596
[M::ha_hist_line]   132: * 1504
[M::ha_hist_line]   133: * 1538
[M::ha_hist_line]   134: * 1551
[M::ha_hist_line]   135: * 1482
[M::ha_hist_line]   136: * 1371
[M::ha_hist_line]   137: * 1301
[M::ha_hist_line]   138: * 1183
[M::ha_hist_line]   139: * 1143
[M::ha_hist_line]   140: * 1203
[M::ha_hist_line]   141: * 1148
[M::ha_hist_line]   142: * 1066
[M::ha_hist_line]   143: * 1117
[M::ha_hist_line]   144: * 950
[M::ha_hist_line]   145: * 889
[M::ha_hist_line]   146: * 920
[M::ha_hist_line]   147: * 796
[M::ha_hist_line]   148: * 804
[M::ha_hist_line]   149: * 711
[M::ha_hist_line]   150: * 779
[M::ha_hist_line]   151: * 749
[M::ha_hist_line]   152: * 707
[M::ha_hist_line]   153: * 794
[M::ha_hist_line]   154: * 737
[M::ha_hist_line]   155: * 805
[M::ha_hist_line]   156: * 806
[M::ha_hist_line]   157: * 735
[M::ha_hist_line]   158: * 808
[M::ha_hist_line]   159: * 724
[M::ha_hist_line]   160: * 688
[M::ha_hist_line]   161: * 714
[M::ha_hist_line]   162: * 782
[M::ha_hist_line]   163: * 739
[M::ha_hist_line]   164: * 642
[M::ha_hist_line]   165: * 592
[M::ha_hist_line]   166: * 603
[M::ha_hist_line]   167: * 642
[M::ha_hist_line]   168: * 607
[M::ha_hist_line]  rest: ******************************************* 48824
[M::ha_analyze_count] left: count[30] = 31980
[M::ha_analyze_count] right: none
[M::ha_pt_gen] peak_hom: 63; peak_het: 30
[M::ha_ct_shrink::646.616*45.13] ==> counted 3625231 distinct minimizer k-mers
[M::ha_pt_gen::] counting in normal mode
[M::yak_count] collected 217854112 minimizers
[M::yak_count] collected 0 minimizers
[M::ha_pt_gen::658.198*44.90] ==> indexed 217690644 positions, counted 3625231 distinct minimizer k-mers
[M::pec::171.140] # bases: 8035090083; # corrected bases: 75738
[M::pec::1.139] # exact o: 32526992; # non-exact o: 831960
[M::ha_assemble::830.515*48.85@17.752GB] ==> corrected reads for round 3
[M::ha_assemble] # bases: 8035090083; # corrected bases: 75738
[M::yak_count] collected 217853961 minimizers
[M::yak_count] collected 0 minimizers
[M::ha_pt_gen::862.811*47.47] ==> counted 3783820 distinct minimizer k-mers
[M::ha_pt_gen] count[4095] = 0 (for sanity check)
[M::ha_analyze_count] lowest: count[5] = 1453
[M::ha_analyze_count] highest: count[63] = 114696
[M::ha_hist_line]     1: ****************************************************************************************************> 158931
[M::ha_hist_line]     2: ** 2275
[M::ha_hist_line]     3: * 1376
[M::ha_hist_line]     4: * 1393
[M::ha_hist_line]     5: * 1453
[M::ha_hist_line]     6: ** 2367
[M::ha_hist_line]     7: *** 3412
[M::ha_hist_line]     8: **** 4992
[M::ha_hist_line]     9: ****** 7230
[M::ha_hist_line]    10: ******** 9597
[M::ha_hist_line]    11: ********** 11410
[M::ha_hist_line]    12: *********** 12811
[M::ha_hist_line]    13: ************ 13507
[M::ha_hist_line]    14: ************ 13503
[M::ha_hist_line]    15: *********** 12821
[M::ha_hist_line]    16: ********** 11313
[M::ha_hist_line]    17: ********* 9927
[M::ha_hist_line]    18: ********* 10222
[M::ha_hist_line]    19: ********* 10599
[M::ha_hist_line]    20: ********* 10830
[M::ha_hist_line]    21: *********** 12112
[M::ha_hist_line]    22: ************ 13899
[M::ha_hist_line]    23: ************** 16065
[M::ha_hist_line]    24: ***************** 19126
[M::ha_hist_line]    25: ******************* 21347
[M::ha_hist_line]    26: ********************* 24290
[M::ha_hist_line]    27: ************************ 27264
[M::ha_hist_line]    28: ************************* 28845
[M::ha_hist_line]    29: *************************** 30554
[M::ha_hist_line]    30: **************************** 31974
[M::ha_hist_line]    31: **************************** 31791
[M::ha_hist_line]    32: *************************** 31012
[M::ha_hist_line]    33: ************************** 30167
[M::ha_hist_line]    34: ************************** 30050
[M::ha_hist_line]    35: ************************* 28291
[M::ha_hist_line]    36: ************************ 27299
[M::ha_hist_line]    37: *********************** 26393
[M::ha_hist_line]    38: *********************** 25926
[M::ha_hist_line]    39: ********************* 24544
[M::ha_hist_line]    40: ********************* 23831
[M::ha_hist_line]    41: ******************** 22966
[M::ha_hist_line]    42: ******************** 23125
[M::ha_hist_line]    43: ********************** 24905
[M::ha_hist_line]    44: *********************** 26903
[M::ha_hist_line]    45: ************************** 29590
[M::ha_hist_line]    46: ***************************** 33134
[M::ha_hist_line]    47: ******************************** 37145
[M::ha_hist_line]    48: ************************************ 40947
[M::ha_hist_line]    49: **************************************** 45341
[M::ha_hist_line]    50: ******************************************** 50583
[M::ha_hist_line]    51: ************************************************** 56941
[M::ha_hist_line]    52: ****************************************************** 62316
[M::ha_hist_line]    53: ************************************************************* 69396
[M::ha_hist_line]    54: ***************************************************************** 74667
[M::ha_hist_line]    55: ********************************************************************** 80226
[M::ha_hist_line]    56: **************************************************************************** 86678
[M::ha_hist_line]    57: ********************************************************************************** 93961
[M::ha_hist_line]    58: ************************************************************************************** 98235
[M::ha_hist_line]    59: ******************************************************************************************* 103944
[M::ha_hist_line]    60: ********************************************************************************************* 107062
[M::ha_hist_line]    61: ************************************************************************************************ 110475
[M::ha_hist_line]    62: ************************************************************************************************** 112765
[M::ha_hist_line]    63: **************************************************************************************************** 114696
[M::ha_hist_line]    64: ************************************************************************************************** 112941
[M::ha_hist_line]    65: ************************************************************************************************* 111426
[M::ha_hist_line]    66: *********************************************************************************************** 108686
[M::ha_hist_line]    67: ******************************************************************************************* 103985
[M::ha_hist_line]    68: ************************************************************************************** 98826
[M::ha_hist_line]    69: ********************************************************************************* 92609
[M::ha_hist_line]    70: ************************************************************************** 85130
[M::ha_hist_line]    71: ******************************************************************* 76592
[M::ha_hist_line]    72: *************************************************************** 71696
[M::ha_hist_line]    73: ********************************************************** 66505
[M::ha_hist_line]    74: ***************************************************** 60379
[M::ha_hist_line]    75: *********************************************** 53423
[M::ha_hist_line]    76: ***************************************** 46485
[M::ha_hist_line]    77: *********************************** 40023
[M::ha_hist_line]    78: ****************************** 34483
[M::ha_hist_line]    79: ************************* 28924
[M::ha_hist_line]    80: ********************* 24100
[M::ha_hist_line]    81: ****************** 20660
[M::ha_hist_line]    82: *************** 17253
[M::ha_hist_line]    83: ************ 13657
[M::ha_hist_line]    84: ********** 11358
[M::ha_hist_line]    85: ******** 9173
[M::ha_hist_line]    86: ******* 7512
[M::ha_hist_line]    87: ***** 6149
[M::ha_hist_line]    88: **** 4938
[M::ha_hist_line]    89: **** 4090
[M::ha_hist_line]    90: *** 3586
[M::ha_hist_line]    91: *** 3061
[M::ha_hist_line]    92: ** 2734
[M::ha_hist_line]    93: ** 2410
[M::ha_hist_line]    94: ** 1886
[M::ha_hist_line]    95: * 1710
[M::ha_hist_line]    96: * 1631
[M::ha_hist_line]    97: ** 1794
[M::ha_hist_line]    98: * 1612
[M::ha_hist_line]    99: * 1489
[M::ha_hist_line]   100: * 1470
[M::ha_hist_line]   101: * 1457
[M::ha_hist_line]   102: * 1573
[M::ha_hist_line]   103: * 1320
[M::ha_hist_line]   104: * 1410
[M::ha_hist_line]   105: * 1368
[M::ha_hist_line]   106: * 1344
[M::ha_hist_line]   107: * 1337
[M::ha_hist_line]   108: * 1313
[M::ha_hist_line]   109: * 1157
[M::ha_hist_line]   110: * 1410
[M::ha_hist_line]   111: * 1244
[M::ha_hist_line]   112: * 1190
[M::ha_hist_line]   113: * 1337
[M::ha_hist_line]   114: * 1447
[M::ha_hist_line]   115: * 1446
[M::ha_hist_line]   116: * 1386
[M::ha_hist_line]   117: * 1361
[M::ha_hist_line]   118: * 1435
[M::ha_hist_line]   119: * 1370
[M::ha_hist_line]   120: * 1464
[M::ha_hist_line]   121: * 1542
[M::ha_hist_line]   122: * 1623
[M::ha_hist_line]   123: * 1652
[M::ha_hist_line]   124: * 1707
[M::ha_hist_line]   125: * 1652
[M::ha_hist_line]   126: ** 1761
[M::ha_hist_line]   127: * 1663
[M::ha_hist_line]   128: * 1665
[M::ha_hist_line]   129: * 1598
[M::ha_hist_line]   130: * 1700
[M::ha_hist_line]   131: * 1590
[M::ha_hist_line]   132: * 1507
[M::ha_hist_line]   133: * 1540
[M::ha_hist_line]   134: * 1552
[M::ha_hist_line]   135: * 1486
[M::ha_hist_line]   136: * 1368
[M::ha_hist_line]   137: * 1300
[M::ha_hist_line]   138: * 1179
[M::ha_hist_line]   139: * 1147
[M::ha_hist_line]   140: * 1185
[M::ha_hist_line]   141: * 1155
[M::ha_hist_line]   142: * 1070
[M::ha_hist_line]   143: * 1113
[M::ha_hist_line]   144: * 956
[M::ha_hist_line]   145: * 892
[M::ha_hist_line]   146: * 926
[M::ha_hist_line]   147: * 798
[M::ha_hist_line]   148: * 798
[M::ha_hist_line]   149: * 718
[M::ha_hist_line]   150: * 768
[M::ha_hist_line]   151: * 750
[M::ha_hist_line]   152: * 708
[M::ha_hist_line]   153: * 792
[M::ha_hist_line]   154: * 738
[M::ha_hist_line]   155: * 806
[M::ha_hist_line]   156: * 807
[M::ha_hist_line]   157: * 736
[M::ha_hist_line]   158: * 800
[M::ha_hist_line]   159: * 724
[M::ha_hist_line]   160: * 687
[M::ha_hist_line]   161: * 720
[M::ha_hist_line]   162: * 779
[M::ha_hist_line]   163: * 738
[M::ha_hist_line]   164: * 641
[M::ha_hist_line]   165: * 596
[M::ha_hist_line]   166: * 600
[M::ha_hist_line]   167: * 648
[M::ha_hist_line]   168: * 600
[M::ha_hist_line]  rest: ******************************************* 48834
[M::ha_analyze_count] left: count[30] = 31974
[M::ha_analyze_count] right: none
[M::ha_pt_gen] peak_hom: 63; peak_het: 30
[M::ha_ct_shrink::862.847*47.47] ==> counted 3624889 distinct minimizer k-mers
[M::ha_pt_gen::] counting in normal mode
[M::yak_count] collected 217853961 minimizers
[M::yak_count] collected 0 minimizers
[M::ha_pt_gen::875.314*47.22] ==> indexed 217695030 positions, counted 3624889 distinct minimizer k-mers
[M::ha_print_ovlp_stat_0] # overlaps: 33380769
[M::ha_print_ovlp_stat_0] # strong overlaps: 22181276
[M::ha_print_ovlp_stat_0] # weak overlaps: 11199493
[M::ha_print_ovlp_stat_0] # exact overlaps: 32499113
[M::ha_print_ovlp_stat_0] # inexact overlaps: 881656
[M::ha_print_ovlp_stat_0] # overlaps without large indels: 33262615
[M::ha_print_ovlp_stat_0] # reverse overlaps: 27659241
[M::ha_print_ovlp_stat_0] # running time: 43.547
[M::ha_assemble::919.605*47.97@17.752GB] ==> found overlaps for the final round
Writing PAF to disk ...... 
PAF has been written.
[M::ha_opt_update_cov_min] updated max_n_chain to 315
Writing reads to disk... 
Reads has been written.
Writing ma_hit_ts to disk... 
ma_hit_ts has been written.
Writing ma_hit_ts to disk... 
ma_hit_ts has been written.
bin files have been written.
[M::purge_dups] homozygous read coverage threshold: 62
[M::purge_dups] purge duplication coverage threshold: 78
[M::ug_ext_gfa::] # tips::4
Writing raw unitig GFA to disk... 
Writing processed unitig GFA to disk... 
[M::purge_dups] homozygous read coverage threshold: 62
[M::purge_dups] purge duplication coverage threshold: 78
[M::mc_solve:: # edges: 22]
[M::mc_solve_core_adv::0.002] ==> Partition
[M::adjust_utg_by_primary] primary contig coverage range: [52, infinity]
Writing hifiasm_op2/chr9_hifi.asm.bp.p_ctg.gfa to disk... 
[M::reduce_hamming_error_adv::0.096] # inserted edges: 4540, # fixed bubbles: 29
[M::adjust_utg_by_trio] primary contig coverage range: [52, infinity]
[M::recall_arcs] # transitive arcs::54
[M::recall_arcs] # new arcs::21924, # old arcs::12486
[M::clean_trio_untig_graph] # adjusted arcs::0
[M::adjust_utg_by_trio] primary contig coverage range: [52, infinity]
[M::recall_arcs] # transitive arcs::62
[M::recall_arcs] # new arcs::21926, # old arcs::12464
[M::clean_trio_untig_graph] # adjusted arcs::0
[M::output_trio_graph_joint] dedup_base::0, miss_base::0
Writing hifiasm_op2/chr9_hifi.asm.bp.hap1.p_ctg.gfa to disk... 
Writing hifiasm_op2/chr9_hifi.asm.bp.hap2.p_ctg.gfa to disk... 
Inconsistency threshold for low-quality regions in BED files: 70%
[M::main] Version: 0.25.0-r726
[M::main] CMD: /pscratch/sd/p/pbarak/tools/hifiasm/./hifiasm -t 64 -o hifiasm_op2/chr9_hifi.asm --write-paf --write-ec temp chr9_HG002_HiFi_preprocess_herro.fastq.gz
[M::main] Real time: 1004.542 sec; CPU: 44218.559 sec; Peak RSS: 17.752 GB


 -->