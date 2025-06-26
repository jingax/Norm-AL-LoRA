import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from peft.tuners.lora import Linear
import json

def print_trainable_parameters(model):
    """
    Prints the total and trainable parameters of the model.
    """
    trainable_params = 0
    total_params = 0
    total_ranks = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'classifier' in name or 'pooler' in name:
            param.requires_grad = True
        if param.requires_grad:
            trainable_params += param.numel()
            total_ranks += 1
            # print(f"Trainable Parameter: {name} | Shape: {param.shape}")
        # else:
        #     print(f"Non-Trainable Parameter: {name} | Shape: {param.shape}")
    print(f"Total Ranks: {total_ranks},")
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Percentage of Trainable Parameters: {100 * trainable_params / total_params:.2f}%")


def plot_layer_ranks(model,target_modules,exp):
    ''' Plot as rank distribution as heatmap post clipping of LoRA weights '''
    number_layer = model.config.num_hidden_layers
    ranks = [{} for _ in range(number_layer)]

    for name, module in model.named_modules():
        if not isinstance(module, Linear):
            continue
        tm = None
        for _ in target_modules:
            if _ in name:
                tm = _
                break
        name = name.split('.')
        for n in name:
            if n.isdigit():
                layer_c = int(n)
                break

        ranks[layer_c][tm] = module.rank_
    with open(f'./outputs/{exp}/ranks.json', 'w') as f:
        json.dump(ranks, f, indent=True) 
    data = np.array([np.array([l[m] for m in target_modules]) for l in ranks]).T
    data = pd.DataFrame(data, index = target_modules,
                  columns = [f"layer {i}" for i in range(number_layer)])
    plt.figure(figsize=(number_layer, 3))
    sns.heatmap(data, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
    plt.title("Rank Distribution chart")
    plt.savefig(f"./outputs/{exp}/LORA.png")
