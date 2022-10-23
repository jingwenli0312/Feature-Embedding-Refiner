# FER-AM
This a repo for developing the Feature Embedding Refiner (FER) based on Attention Model (AM) for TSP and CVRP (work in progress).

# TSP
## Training
```python
python run.py --problem tsp --graph_size 20 --pretrained pretrained/TSP_20/epoch-99.pt --epoch_end 100 --batch_size 512 --epoch_size 10240 --val_size 1000 --eval_batch_size 1000 --T_max 200 --K 6
```

## Test Only
```python
python run.py --eval_only --load_path '{add model to load here}' --problem tsp --graph_size 20 --pretrained pretrained/TSP_20/epoch-99.pt --val_dataset dataset/tsp20_test_seed1234.pkl --val_size 10000 --eval_batch_size 10000 --T_max 200 
```

# CVRP
## Training
```python
python run.py --problem cvrp --graph_size 20 --pretrained pretrained/CVRP_20/epoch-99.pt --epoch_end 100 --batch_size 512 --epoch_size 10240 --val_size 1000 --eval_batch_size 1000 --T_max 200 --K 6
```

## Test only
```python
python run.py --eval_only --load_path '{add model to load here}' --problem cvrp --graph_size 20 --pretrained pretrained/CVRP_20/epoch-99.pt --val_dataset dataset/vrp20_test_seed1234.pkl --val_size 10000 --eval_batch_size 10000 --T_max 200 
```

## Acknowledgements
Thanks to [wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route) for getting me started with the code for the Attention Model (AM).
