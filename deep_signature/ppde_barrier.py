import torch
import torch.nn as nn
import numpy as np
import argparse
import tqdm
import os
import math
import time
from lib.bsde import FBSDE_BlackScholes as FBSDE
from lib.options import Barrier, Lookback
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('../')


def sample_x0(batch_size, dim, device):
    sigma = 0.2
    mu = 0.01
    tau = 0.001
    z = torch.randn(batch_size, dim, device=device)
    x0 = 50*torch.exp((mu-0.5*sigma**2)*tau + sigma*math.sqrt(tau)*z) # lognormal
    return x0


def train(T, n_steps, d, mu, sigma, depth, rnn_hidden, ffn_hidden, max_updates, batch_size, lag, base_dir,
          device, method, **kwargs):

    ts =  torch.linspace(0,T,n_steps+1, device=device)

    # specify type of option
    if kwargs['option'] == 'Barrier':
        option = Barrier(K=kwargs.get('K', 47), B=kwargs.get('B', 53))

    elif kwargs['option'] == 'Lookback':
        option = Lookback(option_type=kwargs.get('option_type', 'call'))

    fbsde = FBSDE(d, mu, sigma, depth, rnn_hidden, ffn_hidden)
    fbsde.to(device)
    optimizer = torch.optim.RMSprop(fbsde.parameters(), lr=0.0005)

    losses = []
    for idx in range(max_updates):
        optimizer.zero_grad()
        x0 = sample_x0(batch_size, d, device)
        if method=="bsde":
            loss, _, _ = fbsde.bsdeint(ts=ts, x0=x0, option=option, lag=lag)
        else:
            loss, _, _ = fbsde.conditional_expectation(ts=ts, x0=x0, option=option, lag=lag)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().item())

        # testing
        if idx%10 == 0:
            with torch.no_grad():
                x0 = torch.ones(5000, d, device=device) # we do monte carlo
                loss, Y, payoff = fbsde.bsdeint(ts=ts,x0=x0,option=option,lag=lag)
                payoff = torch.exp(-mu*ts[-1])*payoff.mean()



    result = {"state":fbsde.state_dict(), "loss":losses}
    torch.save(result, os.path.join(base_dir, "result.pth.tar"))

    return Y[0,0,0].item()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', default='./tmp/', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--d', default=4, type=int)
    parser.add_argument('--max_updates', default=600, type=int)
    parser.add_argument('--ffn_hidden', default=[20,20], nargs="+", type=int)
    parser.add_argument('--rnn_hidden', default=20, type=int)
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--n_steps', default=10, type=int, help="number of steps in time discrretisation")
    parser.add_argument('--lag', default=10, type=int, help="lag in fine time discretisation to create coarse time discretisation")
    parser.add_argument('--mu', default=0.01, type=float, help="risk free rate")
    parser.add_argument('--sigma', default=0.2, type=float, help="risk free rate")
    parser.add_argument('--method', default="bsde", type=str, help="learning method", choices=["bsde","orthogonal"])

    args = parser.parse_args()
    args.n_steps = int(args.T / .01)

    barrier, lookback = True, True

    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda:{}".format(args.device)
    else:
        device= "cpu"

    results_path = os.path.join(args.base_dir, "BS", args.method)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # if barrier option
    if barrier:
        # create output file
        logfile = f"logs/signature_barrier_{args.T}.csv"
        _file = open(logfile, 'w')
        _file.write('d,T,N,run,K,B,y0,runtime\n')

        # for each strike and barrier
        for K in [47]:# [47, 52]:
            for B in [53]: #[46, 53]:
                for run in range(1): #range(10):
                    np.random.seed(run)

                    t_0 = time.time()
                    y0 = train(
                        T=args.T,
                        n_steps=args.n_steps,
                        d=1,
                        mu=args.mu,
                        sigma=args.sigma,
                        depth=args.depth,
                        rnn_hidden=11, #dd+10,
                        ffn_hidden=[11, 11], #[dd+10,dd+10],
                        max_updates=args.max_updates,
                        batch_size=args.batch_size,
                        lag=args.lag,
                        base_dir=results_path,
                        device=device,
                        method=args.method,
                        option='Barrier', #kwargs
                        K=K, #kwargs
                        B=B #kwargs
                    )
                    t_1 = time.time()

                    print(1, args.T, args.n_steps, run, K, B, y0, t_1 - t_0)
                    _file.write('%i, %f, %i, %i, %f, %f, %f, %f\n'
                            % (1, args.T, args.n_steps, run, K, B,
                                y0, t_1 - t_0))
        _file.close()

    # lookback option
    if lookback:
        # create output file
        logfile = f"logs/signature_lookback_{args.T}.csv"
        _file = open(logfile, 'w')
        _file.write('d,T,N,run,option_type,y0,runtime\n')

        # for each option type
        for option_type in ['call']: #['call', 'put']:
            for run in range(1): #range(10):
                np.random.seed(run)

                t_0 = time.time()
                y0 = train(
                    T=args.T,
                    n_steps=args.n_steps,
                    d=1,
                    mu=args.mu,
                    sigma=args.sigma,
                    depth=args.depth,
                    rnn_hidden=11,  # dd+10,
                    ffn_hidden=[11, 11],  # [dd+10,dd+10],
                    max_updates=args.max_updates,
                    batch_size=args.batch_size,
                    lag=args.lag,
                    base_dir=results_path,
                    device=device,
                    method=args.method,
                    option='Lookback',  # kwargs
                    option_type=option_type,  # kwargs
                )
                t_1 = time.time()

                print(1, args.T, args.n_steps, run, option_type, y0, t_1 - t_0)
                _file.write('%i, %f, %i, %i, %s, %f, %f\n'
                            % (1, args.T, args.n_steps, run, option_type,
                               y0, t_1 - t_0))
        _file.close()
