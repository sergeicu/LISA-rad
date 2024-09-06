# val images 
d=/fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/yolov7/coco/images/val2017/
d=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/yolov7/yolov7/coco/images/val2017/



# create link 
cd /fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/LISA
ln -sf $d dataset/coco/train2017
