#!/usr/bin/env python3

import flwr as fl 
from typing import List, Tuple, Dict, Optional  
import torch
from torch.utils.data import DataLoader
import torch.nn as nn  
import utils
import warnings
from argparse import ArgumentParser  

warnings.filterwarnings("ignore")

EXCLUDE_LIST = [
    #"num_batches_tracked",
    #"running",
    #"bn", #FedBN
]
seed = 2022
utils.seed_everything(seed)

        
def get_eval_fn(model, path):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself

    trainset, testset, num_examples = utils.load_isic_data(path)
    trainset, testset, num_examples = utils.load_partition(trainset, testset, num_examples, idx=3, num_partitions=10)  # Use validation set partition 3 for evaluation of the whole model
    
    testloader = DataLoader(testset, batch_size=32, num_workers=4, worker_init_fn=utils.seed_worker, shuffle = False)  
    
    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.NDArrays,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters 
        utils.set_parameters(model, weights, EXCLUDE_LIST) 
        loss, auc, accuracy, f1 = utils.val(model, testloader, nn.BCEWithLogitsLoss(), -1, device) 

        return float(loss), {"accuracy": float(accuracy)}
    return evaluate
    

def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10 
    return {"val_steps": val_steps}



if __name__ == "__main__":

    parser = ArgumentParser()  
    parser.add_argument("--model", type=str, default='efficientnet-b2')
    parser.add_argument("--path_data", type=str, default='/home/gandhi/Project/decentralizedAI_dermatology/archive') 
    parser.add_argument("--tags", type=str, default='Exp 5. FedBN') 

    parser.add_argument(
        "--r", type=int, default=10, help="Number of rounds for the federated training"
    )
    parser.add_argument(
        "--fc",
        type=int,
        default=3,
        help="Min fit clients, min number of clients to be sampled next round",
    )
    parser.add_argument(
        "--ac",
        type=int,
        default=3,
        help="Min available clients, min number of clients that need to connect to the server before training round can start",
    )
    args = parser.parse_args()

    # Setting up GPU for processing or CPU if GPU isn't available
    device = torch.device( f"cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    rounds = int(args.r)
    fc = int(args.fc)
    ac = int(args.ac)

    # Load model for
        # 1. server-side parameter initialization
        # 2. server-side parameter evaluation
    model = utils.load_model(args.model, device).eval() 
    
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit = fc/ac,
        fraction_evaluate = 1,  
        min_fit_clients = fc,
        min_evaluate_clients = 2,  
        min_available_clients = ac,
        evaluate_fn=get_eval_fn(model, args.path_data),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters= fl.common.ndarrays_to_parameters(utils.get_parameters(model, EXCLUDE_LIST)),  
    )

    fl.server.start_server(server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds = rounds), strategy=strategy) 
