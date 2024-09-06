###### rsync the wrist dataset (very large)
ssh ch215616@e3-login.tch.harvard.edu
ssh gpu-b07-0


# rsync all files 
d=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/
cd ~/ww/

cd code/llm/experiments/
# make filee
cd llm/experiments

# copying all contents of llm 
cd ~/ww/code/llm/experiments/yolov7/wrist_fracture_dataset
d=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/
rsync -av $d/yolov7/wrist_fracture_dataset --exclude='dcm/' . 
rsync -av $d/yolov7/wrist_fracture_dataset/dcm  . 
cd ~/ww/code/llm/experiments/yolov7/
rsync -av $d/yolov7/* --exclude='wrist_fracture_dataset/' .
cd ~/ww/code/llm/experiments/
rsync -av $d/ --exclude='yolov7/' --exclude='groundingLMM/' .
cd ~/ww/code/llm/experiments/groundingLMM
rsync -av $d/groundingLMM/* .

# final rsync (copies all)
cd ~/ww/code/llm/experiments/
d=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/
rsync -av $d/* .

# copying all huggingface models 
d=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/huggingface_cache/hub/
cd ~/ww/code/huggingface_cache/hub
rsync -av $d/* .
d=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/huggingface_cache/
cd ~/ww/code/huggingface_cache/hub
rsync -av $d/models--xinlai--LISA-13B-llama2-v1-explanatory .








(tch2_yml) (gamakichi)          ~/w/code/llm/experiments
$ du -sh * 
4.0G    cuda_11.7.0_515.43.04_linux.run
120K    cuda-11-7-0-download-archive?target_os=Linux
5.0G    cuda_12.4.0_550.54.14_linux.run
154G    groundingLMM
19G     LISA
51G     llavamed
134M    MedSAM
968K    ollama
50M     s20240213_llama
6.5G    t
32K     wget-log
246G    yolov7


####################################
# remove some files 
####################################
cd /fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge//mwf_data
[running] rm -i -rf synthetic_data/trained_weights/ &
[X] rm -i -rf synthetic_data/training_data/ &

cd /fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge//s20210930_mult_seg_single_seg_ground_truth_brain_51ms_julia
[X] rm -i -rf generated_data/ &

[X] rm -i -rf /fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/w/s20210831_summary_work_done_w_ic &


cd /fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge//code/mwf/experiments
[running]rm -i -rf s20211120_mwf_generate_10_000_brains/data/s20211120_anima_popvar_mwf_20_std2mean_10_v2_CSF_nonlin_10_000brains &


################################################
# individual dir sizes
################################################

(tch2_yml) (gamakichi)          ~/w/code/llm/experiments
$ du -sh * 
4.0G    cuda_11.7.0_515.43.04_linux.run
120K    cuda-11-7-0-download-archive?target_os=Linux
5.0G    cuda_12.4.0_550.54.14_linux.run
154G    groundingLMM
19G     LISA
51G     llavamed
134M    MedSAM
968K    ollama
50M     s20240213_llama
6.5G    t
32K     wget-log
246G    yolov7

(tch2_yml) (gamakichi)          ~/w/code/diffusion/experiments
$ du -sh * 
6.3G    s20221205_video_diffusion_models
842G    s20221206_diffusers
161M    s20221209_marchenko_pastur
87M     s20230208_competing
590G    s20230211_download_3T_data
68G     s20230606_download_sedated_data
214M    s20230717_raw_data_recon_test
4.4G    s20230719_ground_truth_scan1
144K    s20230919_pds
100G    s20240209_oecorr

(tch2_yml) (gamakichi)          ~/w/code/sd/experiments
$ du -sh * 
119G    s20221028_aversion
8.6G    s20230206_gradio
11G     s20230211_corgis
39G     s20230212_rave
136K    s20230401_booth
27G     s20230617_df
340G    s20230818_planetarium
32G     s20231020_das_haus_stable_xl
9.2G    s20231025_invoke2
1.8G    s20231027_barb
7.3M    s20231120_replicate
du: cannot read directory 's20240121_fonts/cog-quickstart/.cog/tmp/build2695058352': Permission denied
274M    s20240121_fonts
61G     sd





(tch2_yml) (gamakichi)          ~/w/code/ivim_motion/experiments                                                          [35/276]
$ du -sh *                                                                                                                        
40K     ibdCrohnsPatientList - ibdCrohnsPatientList.csv                                                                           
64K     ibdCrohnsPatientList.xlsx                                                                                                 
40K     j.sh                                                                                                                      
120K    OLD_s20240226_wild_bootstrap_for_michelle                                                                                 
40K     processing_extra_case_sila.sh
1.5M    s20211015_starter
448K    s20211022_copy_n_process_cemres_5_patients
4.1G    s20211022_svrtk_week2_plus_ismrm
9.6M    s20211112_extract_slice_timing
2.7M    s20211112_p2s_svrtk_tests
852M    s20211123_build_constructB0_and_svr
1.2G    s20211202_simpleitk
176K    s20211202_try_anima_reg
5.0G    s20220204_build_docker_centos_nlopt
3.1M    s20220210_scim_docker_with_preprocessing
184K    s20220314_docker_ivim_preprocessing
3.3M    s20220322_cpp_simple
5.2M    s20220329_cpp_simple_2
27M     s20220401_verify_cpp_works
288K    s20220408_debug_py_code
64K     s20220427_compute_pointcloud_on_multiple_volumes.sh
2.2M    s20220524_dsvr_split_odd_even
[!!]115G    s20220623_3planesvr
38G     s20220721_3planesvr_v2
1.6M    s20220804_fixing_bsvr
560K    s20220812_m_and_m_subjects
240K    s20220815_mustafas_scan
3.7M    s20220822_axial_bsvr
18G     s20220830_axial_bsvr_more_data
8.5M    s20220830_axial_bsvr_more_data_tess_volunteer_and_lesion
483M    s20220901_svr_vs_ground_truth
483M    s20220901_svr_vs_ground_truth                                                                                      [3/276]
145M    s20220906_simulate_motion
218M    s20220909_fetal_kidney_data
1.7M    s20220913_fetal_brain_for_onur
440K    s20220916_boundaries_for_camilo
6.9M    s20220919_motion_grant
50G     s20220927_binning_registration
1.9G    s20220927_binning_registration_e2
224K    s20221003_constructBval_options
[!!]127G    s20221006_download_hepatoblastoma_cases
18M     s20221018_bootstrap
3.3M    s20221021_download_crohns_cases
74M     s20221110_generate_diffusion_directions
160K    s20221115_onur_lesion
962M    s20230126_cemre
450M    s20230127_cemre
410M    s20230427_ismrm_prep
1.6M    s20230501_process_lesion_kidneys
22M     s20230527_samah
[!!]746G    s20230721_liver_cancers_montage_search
du: cannot access 's20231017_onur_scan_for_dw_grant/22_SB/coil17_w_sneeze/pilottone_binned_3': Permission denied
du: cannot access 's20231017_onur_scan_for_dw_grant/22_SB/coil17_w_sneeze/pilottone_binned_3.mat': Permission denied
du: cannot access 's20231017_onur_scan_for_dw_grant/22_SB/coil17_w_sneeze/pilottone_binned_4.png': Permission denied
du: cannot access 's20231017_onur_scan_for_dw_grant/22_SB/coil17_w_sneeze/pilottone_binned_3.png': Permission denied
du: cannot access 's20231017_onur_scan_for_dw_grant/22_SB/coil17_w_sneeze/pilottone_binned_4': Permission denied
du: cannot access 's20231017_onur_scan_for_dw_grant/22_SB/coil17_w_sneeze/pilottone_binned_4.mat': Permission denied
41G     s20231017_onur_scan_for_dw_grant
[!!]650G    s20231128_create_seg_from_pacs_for_cemre
209M    s20231215_binning_pt_data_for_michelle
122M    s20240223_odd_even
1.6G    s20240226_nonwild_bootstrap_add_noise_onur
8.7G    s20240226_wild_bootstrap_for_michelle
72K     s20240313_svr_docker




(tch2_yml) (gamakichi)          ~/w/code/mwf/experiments                                                                          
$ du -sh *
24G    atlases
1.5M    full-processing-pipeline                                 
40K     gitlistobjectbysize.sh
64K     README.md
5.8M    s20200920-accelerated_example                            
116M    s20201001_MIML_early_experiments                         
1.4M    s20201026-prasloski-julia-anna3-test-retest              
7.8M    s20201209-debugging-julia                                
208K    s20210111-bet-pretraining-synth-examples                 
208K    s20210111-rescaling-real-data                            
208K    s20210114-clinical-volunteer-ALL-algos-example           
240K    s20210114-skull-strip
600K    s20210120-normalization-experiments                      
208K    s20210128-acc-vs-non-acc                                 
1.9M    s20210128-generating-images-for-mwf-grant                
2.0M    s20210129-review-quality-of-recons-for-grant             
72K     s20210131-aligning-flair-to-mwf-maps-example             
45M     s20210204-adding-noise-for-grant                         
27M     s20210205-kspace-images_REMOVE_FILES                     
6.7M    s20210212-midl-debugging-anima-and-epfl                  
2.6M    s20210212-midl-main
208K    s20210218-plot-decay-curves-synth-vs-real                
2.7M    s20210222-skull-strip
601M    s20210419_mrm_mwf_corrections                            
8.6M    s20210512-extract-stats-real-data-MWF-vs-synthetic-data-MWF
696K    s20210514-bet
1.5M    s20210514-check-absolute-values-synthetic-vs-real-data   
4.7M    s20210514-norm-synthdataset-by-voxel-train-ann-and-unet-on-this
4.4M    s20210520-assess-param-stats-on-more-varied-data         
9.1M    s20210524-train-synthmap-more-training-data   
601M    s20210419_mrm_mwf_corrections                                                                                             
8.6M    s20210512-extract-stats-real-data-MWF-vs-synthetic-data-MWF                                                               
696K    s20210514-bet                                                                                                             
1.5M    s20210514-check-absolute-values-synthetic-vs-real-data                                                                    
4.7M    s20210514-norm-synthdataset-by-voxel-train-ann-and-unet-on-this                                                           
4.4M    s20210520-assess-param-stats-on-more-varied-data                                                                          
9.1M    s20210524-train-synthmap-more-training-data                                                                               
208K    s20210525-design-real-data-validation-dataset                                                                             
403M    s20210525-extract-files-for-crl-pipeline-1subject                                                                         
184K    s20210525-t1-to-cpmg-register                                                                                             
880K    s20210526-train-on-more-varied-data                                                                                       
1.5M    s20210607-meeting                                                                                                         
1.4M    s20210607-running-crl-pipeline-from-scratch                                                                               
1.4M    s20210624-extact-params-from-abstract-seg                                                                                 
976K    s20210624-generate-brains-from-pickle-priors                                                                              
15M     s20210629-mwf_to_cpmg_header_align                                                                                        
336K    s20210629-register-mni-to-t1-skull-stripped    
[!!!]451G    s20210630-FULL-pipeline-hammers-to-mwf-prior-stats       
336K    s20210726-NOT_finished_train-miml-on-synth-data          
5.8G    s20210727-generate-more-segmentations                    
187M    s20210729
312K    s20210729-synthesize-train-285rois-multiple-segmentations-multiple-subject-priors
336K    s20210731-opt-to-yaml
74M     s20210731-plot-single-voxel-high-signal-real-vs-synthetic 
121M    s20210805-eval-b1-real-data                              
400K    s20210811-fusing-synthesis-fwdmodel-postprocessing-training-togeter
400K    s20210819_remove_edge_slices                             
208K    s20210819_style_transfer                                 
29M     s20210824_fixing_signal_problem                          
336K    s20210824_more_var
240K    s20210824_remove_edge_slices_earlier                     
336K    s20210824_transfer_learning                              
760K    s20210831_custom_assert
348M    s20210901_self_supervised                                
720K    s20210908-tw
1.4M    s20210909
208K    s20210920_radwarfield_server_full_temp_fix               
6.2M    s20210922_train_MIML_on_our_data                         
450M    s20211003_compare_gamma_gaussian                         
48K     s20211005_log.sh
40K     s20211005_log_summary.sh                                 
160M    s20211005_signal_generation_in_pytorch                   
536K    s20211006_anima_model_for_synth_signal_generation        
440K    s20211007_compare_gamma_gaussian_in_gen_images           
32K     s20211014_contralesions
1.8M    s20211018_calc_accuracy_stats                            
1.9G    s20211018_debug_for_onur                                 
1.7G    s20211021_accuracy_on_synth_data                         
65M     s20211025_fixing_anima

3.5M    s20211025_log
202M    s20211026_log
10M     s20211026_log2
484M    s20211026_log3
[!!!]1.6T    s20211116_anima_signal_gen_gm_wm_split                   
[!!!]2.6T    s20211120_mwf_generate_10_000_brains                     
490M    s20211120_mwf_generate_10_000_brains_code                
288K    s20211129_distributed_training                           
12M     s20211129_log
3.7M    s20211129_log.html
3.1M    s20211129_log.ipynb
5.7G    s20211130_examples
[!!!]201G    s20211201_cycle_gan
4.8M    s20211201_grase_scan
12G     s20220105_process_gt_dataset                             
1.8G    s20220117_generate_data_with_low_GM                      
26M     s20220131_grant_figures
464K    s20220202_ann_retrain_on_synth3                          
1.3G    s20220202_swap_patches
216K    s20220207_debug_miml
72M     s20220215_ms_patient_synthmap                            
19M     s20220301_new_flowchart_figure                           
123M    s20220425_train_w_varied_IE_params                       
30M     s20220729_mwf_0_50_mrm_revision_2                        
208K    s20220811_multicomponentT2toolbox                        
65M     s20221115_mouse
48G     s20230320_media_responses                                
40K     s20230718_dti_ms_for_onur                                
784K    training_scripts
64K     vi_cheatsheet.txt


3.5M    s20211025_log
202M    s20211026_log
10M     s20211026_log2
484M    s20211026_log3
1.6T    s20211116_anima_signal_gen_gm_wm_split
2.6T    s20211120_mwf_generate_10_000_brains
490M    s20211120_mwf_generate_10_000_brains_code
288K    s20211129_distributed_training
12M     s20211129_log
3.7M    s20211129_log.html
3.1M    s20211129_log.ipynb
5.7G    s20211130_examples
201G    s20211201_cycle_gan
4.8M    s20211201_grase_scan
12G     s20220105_process_gt_dataset
1.8G    s20220117_generate_data_with_low_GM
26M     s20220131_grant_figures
464K    s20220202_ann_retrain_on_synth3
1.3G    s20220202_swap_patches
216K    s20220207_debug_miml
72M     s20220215_ms_patient_synthmap
19M     s20220301_new_flowchart_figure
123M    s20220425_train_w_varied_IE_params
30M     s20220729_mwf_0_50_mrm_revision_2
208K    s20220811_multicomponentT2toolbox
65M     s20221115_mouse
48G     s20230320_media_responses
40K     s20230718_dti_ms_for_onur
784K    training_scripts
64K     vi_cheatsheet.txt



################################################
# running rm -rf on gamakichi  - i approximately own 30T of data on this drive 
################################################

I need to remove most of mwf synthetic generations... and clean code 

$ du -sh *                                                                                                                        
16M     advanced_tensorflow                                                                                                       
1.4G    anima                                                                                                                     
1.6G    attention                                                                                                                 
3.6G    bash                                                                                                                      
383M    conda                                                                                                                     
32K     credentials.json                                                                                                          
193M    crkit_lbug                                                                                                                
4.8M    crl-guide-wiki                                                                                                            
2.3M    crl-journal-club
[copying to body - delete after][!!!]1.6T    dce                     
3.1M    DCE_generalTooles
[calculating breakdown][!!!]1.6T    diffusion
12G     docker                  
12M     e2                      
176K    email                   
565M    ext                     
7.8G    fid                     
78M     FileZilla3
14G     fsl                     
240K    fsl_temp                
171G    huggingface_cache
120M    ismrm_2024
97G     ivim                    
1.2G    ivim_fetal
[calculating breakdown][!!!]1.8T    ivim_motion
26M     juliaup
11G     llamagnosis
[calculating breakdown][!]484G    llm
48M     moti
[calculating breakdown][!!!]5.0T    mwf
9.4M    mwf_estimation_external_tools
3.0M    proxy
262M    s20220131_docker_and_git_basics
[calculating breakdown][!!!]649G    sd -> niku 
336K    segment_abdomen
2.4G    svrGAN
16M     svtools
752M    venv
29M     workdir_summaries

~/w/code/llm 



~/w/code
[!!!!!!] dce -> 1.6T -> copying to fileserver external body -> no need to keep all files here... 

~/w/s20211120_mwf_generate_10_000_brains/code
1111111111
[everything is being deleted - nothing significant should remain]

~/w/mwf_data# du -sh * 
390M    bet
11G     experiments
2.8G    ismrm
563M    julia
3.8G    MIML
186G    mwf_maps_all_patients
23G     single_patients

[rm] 11T     code ->>>>>>> NEED TO RESOLVE THIS!!!! 
64K     diffusers.yml
3.0M    Downloads
144K    epg
144K    flip_angle_algorithms
646M    for_cemre
235M    for_onur
32K     fs
40K     home
32K     hosts
4.9G    huggingface
169G    ibdCrohns
176K    intravoxel_algorithms
32K     ipek
139G    IVIM_data
64K     launchrc_init_conda.sh
64K     launchrc_sbatch.sh
64K     launchrc.sh
64K     ldm2.yml
84G     miniconda2
1.4G    miniconda3
160K    motor
[rm] 8.6T    mwf_data -> resolved 
[rm] 56G     _OLD_IVIM_data -> resolved
176K    plot
32K     revision2_copy.yaml
32K     revision2.txt
32K     revision2.yaml
4.5G    s20210419_mrm_mwf_corrections
301M    s20210709-fixing-matlab-ivim-implementation 
838M    s20210804_mwf_scan_processing_for_onur
1.6G    s20210807_dce_matlab_flywheel_gear_for_simon
175M    s20210809-ivim-cpp-build
104K    s20210815_mwf_example_processing_for_onur
10G     s20210815_single_patients_for_onur
[rm] 33G     s20210831_summary_work_done_w_ic -> resolved 
[rm] 265G    s20210930_mult_seg_single_seg_ground_truth_brain_51ms_julia -> resolved 
2.5G    s20211110_docker_svrtk_experiments
272K    s20211112_bahram_code
[rm] 9.6T    s20211120_mwf_generate_10_000_brains -> resolved 
276M    s20220209_git_by_example
32K     serge
3.5G    sergeDCE
8.7G    soft
64K     tch2_yml_20211208.yml
32K     tch2_yml_copy_29mar2023_copy.yml
32K     tch2_yml_copy_29mar2023.yml
32K     trained_weights
1.5M    trash
4.3M    venv
