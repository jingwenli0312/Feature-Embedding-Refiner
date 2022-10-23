# FER-POMO
This a repo for developing the Feature Embedding Refiner (FER) based on Policy Optimization with Multiple Optima (POMO) for TSP and CVRP (work in progress).

# TSP
## Training
```python
python run.py --problem tsp --graph_size 20 --pretrained pretrained/Saved_TSP20_Model/ACTOR_state_dic.pt --epoch_end 100 --batch_size 100 --epoch_size 1000 --val_size 1000 --eval_batch_size 1000 --T_max 200 --K 6
```

## Test Only
### Without Augment
```python
python run.py --eval_only --load_path '{add model to load here}' --problem tsp --graph_size 20 --pretrained pretrained/Saved_TSP20_Model/ACTOR_state_dic.pt --val_dataset dataset/tsp20_test_seed1234.pkl --val_size 10000 --eval_batch_size 10000 --T_max 200 
```
### Without Augment
```python
python run.py --eval_only --load_path '{add model to load here}' --problem tsp --graph_size 20 --pretrained pretrained/Saved_TSP20_Model/ACTOR_state_dic.pt --val_dataset dataset/tsp20_test_seed1234.pkl --val_size 10000 --eval_batch_size 10000 --T_max 200 --augment
```

# CVRP
## Training
```python
python run.py --problem cvrp --graph_size 20 --pretrained pretrained/Saved_CVRP20_Model/ACTOR_state_dic.pt --epoch_end 100 --batch_size 100 --epoch_size 1000 --val_size 1000 --eval_batch_size 1000 --T_max 200 --K 6
```

## Test only
### Without Augment
```python
python run.py --eval_only --load_path '{add model to load here}' --problem cvrp --graph_size 20 --pretrained pretrained/Saved_TSP20_Model/ACTOR_state_dic.pt --val_dataset dataset/vrp20_test_seed1234.pkl --val_size 10000 --eval_batch_size 10000 --T_max 200 
```
### With Augment
```python
python run.py --eval_only --load_path '{add model to load here}' --problem cvrp --graph_size 20 --pretrained pretrained/Saved_TSP20_Model/ACTOR_state_dic.pt --val_dataset dataset/vrp20_test_seed1234.pkl --val_size 10000 --eval_batch_size 10000 --T_max 200 --augment
```

## Acknowledgements
Thanks to [yd-kwon/POMO](https://github.com/yd-kwon/POMO) for getting me started with the code for the Policy Optimization with Multiple Optima (POMO).
