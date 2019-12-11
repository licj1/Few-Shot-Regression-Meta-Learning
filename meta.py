from __future__ import division, print_function, absolute_import

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np 
import matplotlib.pyplot as plt
from copy import deepcopy

FLAGS = argparse.ArgumentParser()

# Parameters
FLAGS.add_argument('--mode', type=str, choices=['maml', 'reptile'])
FLAGS.add_argument('--n_shot', type=int, 
    help= "How many samples points to regress on while training.")
FLAGS.add_argument('--train_func', type=str, choices=['sin', 'cos', 'linear'], default='sin',
    help = "Base function you want to use for traning")
FLAGS.add_argument('--iterations', type=int, default=20000)
FLAGS.add_argument('--outer_step_size', type=float, default=0.1)
FLAGS.add_argument('--inner_step_size', type=float, default=0.02)
FLAGS.add_argument('--inner_grad_steps', type=int, default=1)
FLAGS.add_argument('--eval_grad_steps', type=int, default=10)    
FLAGS.add_argument('--eval_iters', type=int, default=5, 
    help='How many testing samples of k different shots you want to run')
FLAGS.add_argument('--logdir', type=str, default="runs", 
    help="TensorBoard Logging")
FLAGS.add_argument('--seed', type=int, default=1)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

window_size = 8
prediction_range = 5
x_all = np.linspace(-prediction_range, prediction_range, num = 50)

def sin(values, amplitude, phase, vertical_shift):
    return amplitude * np.sin(values + phase) + vertical_shift

def cos(values, amplitude, phase, vertical_shift):
    return amplitude * np.cos(values + phase) + vertical_shift

def linear(values, slope, phase, vertical_shift):
    return slope * (values + phase) + vertical_shift
    
def generate(func):
    amplitude = np.random.uniform(low = 0.1, high = 5)
    phase = np.random.uniform(low = 0, high = 2*np.pi)
    vertical_shift = np.random.uniform(low = 0, high = 3) 
    return func(x_all, amplitude, phase, vertical_shift), [amplitude, phase, vertical_shift]

def select_points(wave, k):
    random_points = np.random.choice(np.arange(50), k,replace=False)[:, None]
    return x_all[random_points], wave[random_points] 

def plot_tensorboard(writer, y_eval, pred, k, n , learner, wave_name='SinWave'):
    for j in range(len(y_eval)):
        writer.add_scalars('Test_Run_{}/{}/{}/{}_points_sampled'.format(n,
                learner,wave_name,str(k)), 
            {'Original Function': y_eval[j], 'Pretrained': pred['pred'][0][j][0], 
                'Gradient_Step_{}'.format(len(pred['pred'])-1): pred['pred'][-1][j][0]}, j)  

class Meta_Learning:
    def __init__(self, model, writer):
        self.model = model.to(device)
        self.writer = writer
    
    def train_maml(self, func, k, iterations, outer_step_size, inner_step_size, 
        inner_gradient_steps, tasks=5):
        loss = 0
        batches = 0
        for iteration in range(iterations):
            init_weights = deepcopy(self.model.state_dict())
            y_all , _ = generate(func)
            x_test,y_test = select_points(y_all, k)
            meta_params = {}
            for task in range(tasks): 
                # sample for meta-update
                x,y = select_points(y_all, k)
                for grad_step in range(inner_gradient_steps):
                    loss_base = self.train_loss(x,y)
                    loss_base.backward()
                    for param in self.model.parameters():
                        param.data -= inner_step_size * param.grad.data
                loss_meta = self.train_loss(x_test, y_test)
                loss_meta.backward()
                for name,param in self.model.named_parameters():
                    if(task == 0):
                        meta_params[name] =  param.grad.data
                    else:
                        meta_params[name] += param.grad.data
                loss += loss_meta.cpu().data.numpy()
                batches += 1
                self.model.load_state_dict(init_weights)
            learning_rate = outer_step_size * (1 - iteration/iterations)
            self.model.load_state_dict({name: init_weights[name] - 
                learning_rate/tasks * meta_params[name] for name in init_weights})
            self.writer.add_scalar('MAML/Training/Loss/', loss/batches, iteration)
            if(iteration % 1000 == 0):
                pred = self.predict(x_all[:,None])
                for i in range(len(x_all)):
                    self.writer.add_scalars('MAML/Training/PreTrained_Wave', 
                        {'pretrain_wave_{}'.format(iteration/1000): pred[i][0]},i)  

    def train_reptile(self, func, k, iterations, outer_step_size, inner_step_size, 
        inner_gradient_steps):
        loss = 0
        batches=0
        for iteration in range(iterations):
            init_weights = deepcopy(self.model.state_dict())
            y_all , _ = generate(func)
            for j in range(inner_gradient_steps):
                random_order = np.random.permutation(len(x_all))
                for start in range(0,len(x_all), k):
                    indicies = random_order[start: start + k][:, None]
                    loss_base = self.train_loss(x_all[indicies], y_all[indicies])
                    loss_base.backward()
                    for param in self.model.parameters():
                        param.data -= inner_step_size * param.grad.data
                    loss += loss_base.cpu().data.numpy()
                    batches += 1
            learning_rate = outer_step_size * (1 - iteration/iterations)
            curr_weights = self.model.state_dict()
            self.model.load_state_dict({name: (init_weights[name] + learning_rate * 
                (curr_weights[name] - init_weights[name])) for name in curr_weights})
            self.writer.add_scalar('Reptile/Training/Loss/', loss/batches, iteration)
            if(iteration % 1000 == 0):
                pred = self.predict(x_all[:,None])
                for i in range(len(x_all)):
                    self.writer.add_scalars('Reptile/Training/PreTrained_Wave', 
                        {'pretrain_wave_{}'.format(iteration/1000): pred[i][0]},i)

    def train_loss(self, x, y):
        x = torch.tensor(x, dtype=torch.float32, device = device)
        y = torch.tensor(y, dtype=torch.float32, device = device)
        self.model.zero_grad()
        out = self.model(x)
        loss = (out - y).pow(2).mean()
        return loss

    def eval(self, y_all, k, gradient_steps=10, inner_step_size=0.02):
        x_p,y_p = select_points(y_all, k)
        pred = [self.predict(x_all[:,None])]
        meta_weights = deepcopy(self.model.state_dict())
        for i in range(gradient_steps):
            loss_base = self.train_loss(x_p,y_p)
            loss_base.backward()
            for param in self.model.parameters():
                param.data -= inner_step_size * param.grad.data
            pred.append(self.predict(x_all[:, None]))
        loss = np.power(pred[-1] - y_all,2).mean()
        self.model.load_state_dict(meta_weights)
        return {"pred": pred, "sampled_points":(x_p, y_p)}

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=device)
        return self.model(x).cpu().data.numpy()

class Meta_Wave(nn.Module):
    def __init__(self, units):
        super(Meta_Wave, self).__init__()
        self.inp = nn.Linear(1, units)
        self.layer1 = nn.Linear(units,units)
        self.out = nn.Linear(units, 1)

    def forward(self,x):
        x = torch.tanh(self.inp(x))
        x = torch.tanh(self.layer1(x))
        output = self.out(x)
        return output

def main():
    
    args, unparsed = FLAGS.parse_known_args()
    if len(unparsed) != 0:
        raise NameError("Argument {} not recognized".format(unparsed))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    k = args.n_shot
    iterations = args.iterations
    writer = SummaryWriter(args.logdir)
    if(args.train_func == 'sin'):
        t_f = sin 
    elif(args.train_func == 'cos'):
        t_f = cos
    else:
        t_f = linear

    model = Meta_Wave(64)
    meta = Meta_Learning(model, writer)
    if(args.mode == 'maml'):
        meta.train_maml(t_f, k, iterations, args.outer_step_size, args.inner_step_size,
            args.inner_grad_steps)
        learner = 'maml'
    else:
        meta.train_reptile(t_f ,k, iterations, args.outer_step_size, args.inner_step_size,
            args.inner_grad_steps)
        learner = 'reptile'

    # eval
    eval_iters = args.eval_iters
    gradient_steps = args.eval_grad_steps
    inner_step_size = 0.01
    func_name = ['SinWave', 'CosWave', 'Linear']
    funcs = [sin, cos, linear]
    for n in range(eval_iters):
        for f,name in zip(funcs, func_name):
            y_eval, _  = generate(f)
            for sample in [5,10,20]:
                pred = meta.eval(y_eval, sample, gradient_steps, inner_step_size)
                plot_tensorboard(writer, y_eval, pred, sample, n, learner, wave_name=name)

    writer.close()

if __name__ == "__main__":
    main()