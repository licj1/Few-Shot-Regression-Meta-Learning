## Few Shot Regression of Periodic and Basic Polynomial Functions using MAML and Reptile Methods

### Usage instructions
#### Default Parameters:
```bash
python meta.py --mode=maml --n_shot=10 --train_func=sin --iterations=20000 --outer_step_size=0.05 --inner_step_size=0.02 --inner_grad_steps=1 --eval_grad_steps=10 --eval_iters=5 --logdir=logs/maml --seed=1
```
#### 5-Shot Cos Function MAML 
```bash
python meta.py --mode=maml --n_shot=5 --train_func=cos --logdir=logs/maml_5
```
#### 10-Shot Sin Function Reptile
``` bash
python meta.py --mode=reptile --n_shot=10 --train_func=sin --logdir=logs/reptile_20
```
#### 20-Shot Linear Function Reptile
``` bash
python meta.py --mode=reptile --n_shot=20 --train_func=linear --logdir=logs/reptile_linear_20
```

#### TensorBoard Logs Viewing 
###### inclued logs of Maml and Reptile with default parameters
``` bash
tensorboard --logdir=logs/maml
```
