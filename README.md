# pixelNeRF: Neural Radiance Fields from One or Few Images

### llff
python eval/eval_custom_splits_llff.py -D rs_dtu_4 --split val -n dtu -P '22 25 28' -O eval_out/testllff --gpu_id='1' --write_compare --dataset_format llff --datadir rs_dtu_4/nerf_llff_data 

### dtu
python eval/eval_custom_splits.py -D rs_dtu_4 --split val -n dtu -P '22 25 28' -O eval_out/testdtu --gpu_id='1' --write_compare
