export CUDA_VISIBLE_DEVICES=3,4
export TORCH_DISTRIBUTED_DEBUG="DETAIL"
ckpt=./ckpt/finetuned.pth

# Sound source
dataset=mrsdrama
audio_path_root=./data/gt
output_dir=./outputs/gt/npy
log_dir=./outputs/gt/log

# extract embeddings from SpatialAST

mkdir -p $output_dir

python -m torch.distributed.launch \
    --nproc_per_node=1 --use_env main.py \
    --log_dir $log_dir --output_dir $output_dir --finetune $ckpt \
    --model build_EVA --dataset $dataset \
    --audio_path_root $audio_path_root \
    --audioset_train $audioset_train_json --audioset_eval $audioset_eval_json \
    --nb_classes 355 --batch_size 32 --num_workers 4 \
    --audio_normalize --eval

audio_path_root=./data/infer
output_dir=./outputs/infer/npy

mkdir -p $output_dir

python -m torch.distributed.launch \
    --nproc_per_node=1 --use_env main.py \
    --log_dir $log_dir --output_dir $output_dir --finetune $ckpt \
    --model build_EVA --dataset $dataset \
    --audio_path_root $audio_path_root \
    --audioset_train $audioset_train_json --audioset_eval $audioset_eval_json \
    --nb_classes 355 --batch_size 32 --num_workers 4 \
    --audio_normalize --eval

# extract IPD and ILD

python ./evaluate/gen_ipd_ild.py \
    --audio-dir $audio_path_root \
    --output-dir $output_dir

audio_path_root=./data/gt
output_dir=./outputs/gt/npy

python ./evaluate/gen_ipd_ild.py \
    --audio-dir $audio_path_root
    --output-dir $output_dir

# evaluate

python ./evaluate/evluate.py