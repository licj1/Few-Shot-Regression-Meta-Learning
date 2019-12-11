## Few Shot Regression of Periodic and Basic Polynomial Functions using MAML and Reptile Methods

Humans adapt to new environments and information with only a few samples. Future deep-learning models will adapt and learn with only a few samples. MAML and Reptile are gradient based meta-learning algorithms which initialize parameters that are highly sensitive to changes in tasks. We examine the evolution of changes in the sensitive parameters over time and its relative performance to unseen takes.  

### Formulation of Task: 
###### Added an additional transformation to the original formulation of the regression task from paper Model Agnostic Meta Learning.  
Each task involves regressing a periodic or polynomial function. We sample amplitude/slope from the [0.1, 5], phase shift from [0, 2pi] and vertical shift [0, 3], for each periodic or polynomial task respectively. During training and testing we uniformly sample k points from [-5, 5]. 

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
##### inclued logs of Maml and Reptile with default parameters
``` bash
tensorboard --logdir=logs/maml
```
#### Retile Traning Loss and Pre-Trainied Wave every 1000 iterations
![Alt Text](https://github.com/vinit97/Few-Shot-Regression-Meta-Learning/blob/master/logs/train_reptile.png)
#### Example of Test Run
![Alt Text](https://github.com/vinit97/Few-Shot-Regression-Meta-Learning/blob/master/logs/test_reptile.png)
