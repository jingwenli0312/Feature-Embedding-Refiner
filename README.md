# Learning Feature Embedding Refiner for Solving Vehicle Routing Problems

Based on pre-traiend neural construction models such as Attention model (AM) and Policy Optimization with Multiple Optima (POMO), feature embedding refiner (FER) learns to iteratively boost the performance of construction models by adding a refiner between the encoder and decoder to better synergize them.

## Paper
For more details, please see our paper： Jingwen Li, Yining Ma, Zhiguang Cao, Yaoxin Wu, Wen Song, Jie Zhang, Yeow Meng Chee. [Learning Feature Embedding Refiner for Solving Vehicle Routing Problems](https://ieeexplore.ieee.org/document/10160045). IEEE Transactions on Neural Network and Learning Systems, 2023. If this code is useful for your work, please cite our paper,

```
@article{li2023fer,
  title={Learning Feature Embedding Refiner for Solving Vehicle Routing Problems},
  author={Li, Jingwen and Ma, Yining and Cao, Zhiguang and Wu, Yaoxin and Song, Wen and Zhang, Jie and Chee, Yeow Meng},
  journal={IEEE Transactions on Neural Network and Learning Systems (Early Access)},
  year={2023},
  publisher={IEEE}，
  doi={10.1109/TNNLS.2023.3285077}
}
```

# dependencies
* Python>=3.7
* PyTorch>=1.6
* numpy
* tqdm
* tensorboard_logger

## Details
For more details, please see the FER-AM and FER-POMO for solving TSP and CVRP, respectively.
