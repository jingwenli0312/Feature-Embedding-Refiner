# PDP-improve
This a repo for developing the improvement-based DRL solver for PDP (work in progress).

![pdp](./outputs/pdp_0.gif) ![pdp](./outputs/pdp_1.gif)

# dependencies
* Python>=3.6
* PyTorch>=1.1
* numpy
* tqdm
* cv2
* tensorboard_logger

## One more thing
For the exception below from package tensorboard_logger,
```python
AttributeError: module 'scipy.misc' has no attribute 'toimage'
```
Please refer to [issue #27](https://github.com/TeamHG-Memex/tensorboard_logger/issues/27) to fix it.

# TSP
## Training
```python
CUDA_VISIBLE_DEVICES=5,6 python run.py --problem tsp --graph_size 20 --seed 1234 --n_epochs 100 --batch_size 512 --epoch_size 5120 --val_size 1000 --eval_batch_size 1000 --val_dataset './datasets/tsp_20_10000.pkl' --no_assert --run_name training
```

## Debug
```python
CUDA_VISIBLE_DEVICES=5,6 python run.py --problem tsp --graph_size 20 --seed 1234 --n_epochs 4  --epoch_size 4 --batch_size 2 --val_size 4 --eval_batch_size 4 --run_name debug --val_dataset './datasets/tsp_20_10000.pkl'

```
# PDP
## Training
```python
CUDA_VISIBLE_DEVICES=6,5 python run.py --problem pdp --graph_size 20 --seed 1234 --n_epochs 100 --batch_size 512 --epoch_size 5120 --val_size 1000 --eval_batch_size 1000 --no_assert --run_name pdp_first --val_dataset './datasets/pdp_20_10000.pkl'
```

## Debug
```python
CUDA_VISIBLE_DEVICES=5,6 python run.py --problem pdp --graph_size 20 --seed 1234 --n_epochs 4  --epoch_size 4 --batch_size 2 --val_size 4 --eval_batch_size 4 --run_name debug --val_dataset './datasets/pdp_20_10000.pkl'

```

# Test only
```python
--eval_only --load_path '{add model to load here}'
```

# Acknowledgements
The code and the framework are based on the repo [wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route) and the [codes](https://drive.google.com/file/d/1cN61NAHns6cu8TrEsnURIWmbaGe98ffG/view?usp=sharing) of the paper [Wu et. al., *Learning Improvement Heuristics for Solving Routing Problems*, arXiv, 2019](https://arxiv.org/abs/1912.05784).