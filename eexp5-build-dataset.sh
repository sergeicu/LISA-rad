# dataset pipeline: 
/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/pipeline.sh

# report pipeline: 
/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/ollama/ollama_llama3.sh # uses llama3.1-7B model with basic prompt 

# tasks: 
- rerun report pipeline - generate Q&A with Andys knowledge 
- create semantic segmentations - see below ... 
- create symlinks only to those that exist 

python create_symlinks.py <source_folder> <target_folder> [--remove-existing] [--skip-existing]

# 


# build our own dataset: 
- prepare medical reports + Q&A + images (all have to be in the same place)


# initial prep was here...



############################################
#### CONVERSATION DESIGN 
############################################
in train_ds.py 
input_dict

In [7]: input_dict['questions_list']
Out[7]: 
[['<image>\nPlease segment the metal in this image.',
  '<image>\nCan you segment the tree in this image?',
  '<image>\nPlease segment the roof in this image.'],
 ['<image>\nCan you segment the cake in this image?',
  '<image>\nWhat is paper in this image? Please output segmentation mask.',
  '<image>\nCan you segment the person in this image?']]

In [6]: input_dict['conversation_list']
Out[6]: 
["A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <im_start><image><im_end>\nPlease segment the metal in this image. ASSISTANT: Sure, it is [SEG].</s>",
 "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <im_start><image><im_end>\nCan you segment the tree in this image? ASSISTANT: [SEG].</s>",
 "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <im_start><image><im_end>\nPlease segment the roof in this image. ASSISTANT: It is [SEG].</s>",
 "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <im_start><image><im_end>\nCan you segment the cake in this image? ASSISTANT: Sure, [SEG].</s>",
 "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <im_start><image><im_end>\nWhat is paper in this image? Please output segmentation mask. ASSISTANT: It is [SEG].</s>",
 "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <im_start><image><im_end>\nCan you segment the person in this image? ASSISTANT: Sure, the segmentation result is [SEG].</s>"]


 In [9]: input_dict['image_paths']
Out[9]: 
['./dataset/coco/train2017/000000224724.jpg',
 './dataset/coco/train2017/000000321214.jpg']

In [11]: type(input_dict['images'])
Out[11]: torch.Tensor


In [13]: input_dict['images'].size()
Out[13]: torch.Size([2, 3, 1024, 1024])

In [15]: input_dict['images_clip'].size()
Out[15]: torch.Size([2, 3, 224, 224])

In [18]: input_dict['input_ids'].size()
Out[18]: torch.Size([6, 61])

In [19]: input_dict['input_ids'][0,:]
Out[19]: 
tensor([    1,   319, 13563,  1546,   263, 12758,  5199,   322,   385, 23116,
        21082, 20255, 29889,   450, 20255,  4076,  8444, 29892, 13173, 29892,
          322,  1248,   568,  6089,   304,   278,  5199, 29915, 29879,  5155,
        29889,  3148,  1001, 29901, 32001,  -200, 32002,  3529, 10768,   278,
        11915,   297,   445,  1967, 29889,   319,  1799,  9047, 13566, 29901,
        18585, 29892,   372,   338, 32003,   869,     2,     0,     0,     0,
            0], device='cuda:0')


In [22]: input_dict['labels'].size()
Out[22]: torch.Size([6, 61])

In [21]: input_dict['labels'][0,:]
Out[21]: 
tensor([ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        18585, 29892,   372,   338, 32003,   869,     2,  -100,  -100,  -100,
         -100], device='cuda:0')


In [28]: input_dict['masks_list'][0].size()
Out[28]: torch.Size([3, 640, 480])


In [30]: input_dict['attention_masks'].size()
Out[30]: torch.Size([6, 61])



In [31]: input_dict['label_list'][0].size()
Out[31]: torch.Size([640, 480])

In [33]: input_dict['resize_list'][0]
Out[33]: [1024, 768]

In [35]: input_dict['offset']
Out[35]: tensor([0, 3, 6], device='cuda:0')

In [36]: input_dict['sampled_classes_list']
Out[36]: [['metal', 'tree', 'roof'], ['cake', 'paper', 'person']]

In [37]: input_dict['inference']
Out[37]: False


############################################
#### My design
############################################
This file answers
1. Where is the location of all the phrases. 
2. How do i construct a mechanism for doing this?  Dataset class 
3. We need to pick only the images that have at least one bounding box - is important... 
4. Can I turn bounding boxes into masks... (important)
5. Can I turn SAM into mechanism to predict bounding box coordinates (where are bounding)

############################################
#### Setting up locations 
############################################
1. images (no segmentations)
/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/pngs

2. reports (old - v8)
/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/reports_llama3_1_8b_v8
example is this - 26909790_report_v8.json
  "formatted_response": {
    "question": "what abnormalities are visible in this wrist x-ray?",
    "answer": "the x-ray shows transverse fractures of the right distal radius and ulna with near anatomic alignment indicating early healing. there is also evidence of disuse osteoporosis.",
    "grounding_phrase": "right distal radial and ulnar metadiaphyseal transverse fractures"
  }

3. reports new (but not finished yet because i did not run llama on them - do it later today) 
/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/

4. yolo inferrence results

    # labels 
    
    /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/results/v2/run1/labels
        #/fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/results/v2/run1/labels/

    # images (predicted bounding boxes)
    /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/results/v2/run1/

    # script that did this - 
    /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/create_wrist_fracture_dataset/yolov7-grazped-inferrence.sh

    # bad (do not use this)
    /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/yolov7/runs/detect/exp13 
    note that some of them may not have any boxes... important... where are the coordinates? 

5. create semantic segmentations (no need for bounding box files since these are in text files)

    # script that turns predicted coordinates to bounding boxes 
    d=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/yolov7/
    python $d/draw_bounding_boxes_v2.py  # need to convert this to work on ALL our images (with coordinates)
                                            # and then convert to semantic segmentation

    # build semantic segmentations for LISA 
    d=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/
    txt_folder=$d/results/v2/run1/labels/
    reference_folder=$d/results/v2/run1/
    input_folder=$d/pngs/
    output_folder=$d/pngs_semantic/ # all classes only
    output_reference_folder=$d/pngs_semantic_comps/
    python batch_draw_semantic_segmentations.py $txt_folder $reference_folder $input_folder $output_folder $output_reference_folder
    output_folder=$d/pngs_semantic_f/ # fracture only
    output_reference_folder=$d/pngs_semantic_comps_f/
    python batch_draw_semantic_segmentations.py $txt_folder $reference_folder $input_folder $output_folder $output_reference_folder --fracture_only
    

    # create symlinks only to those labels that exist 
    source_folder=$d/pngs_semantic_f/ # fracture only 
    target_folder=$d/pngs_semantic_f_symlinks/ # only selects files that have labels
    python create_symlinks.py $source_folder $target_folder #[--remove-existing] [--skip-existing]    
    source_folder=$d/pngs_semantic/ # all classes
    target_folder=$d/pngs_semantic_symlinks/ # only selects files that have labels
    python create_symlinks.py $source_folder $target_folder #[--remove-existing] [--skip-existing]    

    # count files
        source_folder=$d/pngs_semantic_f/ # all classes
        target_folder=$d/pngs_semantic_f_symlinks/ # only selects files that have labels
        ls $source_folder | wc -l # 26107 # this means we only made 26,000 predictions (out of almost 40,000)
        ls $target_folder | wc -l  # 17535 # this  # this means we lost almost 9,000 files since they did not have any fractures 


        source_folder=$d/pngs_semantic/ # all classes
        target_folder=$d/pngs_semantic_symlinks/ # only selects files that have labels
        ls $source_folder | wc -l # 26107
        ls $target_folder | wc -l  # 26107



    # create symlinks for input files for which there is a corresponding label 
    inputfolder=$d/pngs/ 
    inputfolder_symlink=$d/pngs_f_symlink/ # fracture only 
    labelsfolder=$d/pngs_semantic_f_symlinks/
    python create_input_symlinks.py $inputfolder $labelsfolder $inputfolder_symlink
    inputfolder_symlink=$d/pngs_symlink/ # all classes
    labelsfolder=$d/pngs_semantic_symlinks/
    python create_input_symlinks.py $inputfolder $labelsfolder $inputfolder_symlink    


    # turns out we did not finish almost 10,000 images
        inputfolder=$d/pngs/ 
        inputfolder_symlink=$d/pngs_f_symlink/ # fracture only 
        ls $inputfolder | wc -l  # 38074 -> this the total number of images we should be getting 
        ls $inputfolder_symlink | wc -l  # 17535 -> this is the total number of images after discounting for predictions (10,000) and missing fracture labels (9,000)



    # link the folders to input 
    d=/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/yolov7/wrist_fracture_dataset/
    inputfolder_symlink=$d/pngs_f_symlink/
    cd ~/ww/code/llm/experiments/LISA/dataset/bchwrist_images
    ln -sf $inputfolder_symlink train
    cd ~/ww/code/llm/experiments/LISA/dataset/bchwrist_labels
    target_folder_symlink=$d/pngs_semantic_f_symlinks/ # only selects files that have labels
    ln -sf $target_folder_symlink train

    # then rewrite the dataset generation class [already did this]
    


    


