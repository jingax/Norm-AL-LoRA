import torch
from transformers import TrainerCallback
from peft.tuners.lora import Linear
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def mask_clipped(model,reset):
    '''
    In our implementation we do not save the rank_ attribute (new clipped rank) of the adapter
    When a the adaptor is reloaded, the clipped weights are enabled again as rank_ is intialised with r (full rank)
    To fix that we zero out all the clipped weights before saving
    This method zeros all weights with rank > rank_

    args:
    model : LoRA injected model
    reset : (yes/no) After clipping should the weights be reintialized. This is done to removed the condition of decreasing norms which is not required after clipping.
            By defualt it is set `no` 
    '''
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, Linear):
                for adpt in module.active_adapters:
                    if reset == 'yes':
                        print(f"Reseting module {name}")
                        module.reset_lora_parameters(adpt,True)
                        module.hooked = False
                    module.apply_mask(adpt)

def update_scaling(model):
    '''
    scaling or alpha can be upadted to after clipping of LoRA weights
    By default it is disabled as no improvement was observed in experiments
    To enable it call this method in mask_clipped method
    '''
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, Linear):
                for adpt in module.active_adapters:
                    module.update_scaling(adpt)

def display_ranks(model,max_rank):
    '''
    Print the new clipped ranks as list and final average rank of the LoRA adaptor
    '''
    with torch.no_grad():
        ranks = []
        for name, module in model.named_modules():
            if isinstance(module, Linear):
                norms = []
                for rank in range(max_rank):
                    last_vector_A = module.lora_A['default'].weight[rank,:]
                    last_vector_B = module.lora_B['default'].weight[:,rank]
                    normA = torch.norm(last_vector_A, p=2)
                    normB = torch.norm(last_vector_B, p=2)
                
                    norms.append(normA+normB)
                for adpt in module.active_adapters:
                    ranks.append(module.rank_)
                norms = []
                for rank in range(module.rank_):
                    last_vector_A = module.lora_A['default'].weight[rank,:]
                    last_vector_B = module.lora_B['default'].weight[:,rank]
                    normA = torch.norm(last_vector_A, p=2)
                    normB = torch.norm(last_vector_B, p=2)
                    
                    norms.append(normA+normB)
                
        print(f"Final rank list :{ranks}")
        print(f"Final average rank :{sum(ranks)/len(ranks)}")

def clip_rank_threshold(model,k):
    '''
    Norm threshold based clipping
    Returns a Dict() of clipped ranks for each layer as {layer_name: clipped_rank}
    Retains all rank vectos where norm is greater than the threshold k/100 (requires 0 < k < 100)
    To use this add args `--clip_strategy zeros` in the cmd
    '''
    rank_dict = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, Linear):
                rank_dict[name] = module.rank_
                for rank in range(module.rank_):
                    last_vector_A = module.lora_A['default'].weight[rank,:]
                    last_vector_B = module.lora_B['default'].weight[:,rank]
                    normA = torch.norm(last_vector_A, p=2)
                    normB = torch.norm(last_vector_B, p=2)
                    
                    # first norm required for normalization
                    last_vector_A = module.lora_A['default'].weight[0,:]
                    last_vector_B = module.lora_B['default'].weight[:,0]
                    normA0 = torch.norm(last_vector_A, p=2)
                    normB0 = torch.norm(last_vector_B, p=2)
                    if min(1,(normA+normB)/(normA0+normB0)) < k / 100: #retain all ranks with norm > k/100
                        module.rank_ = rank
                        rank_dict[name] = rank
                        break
    return rank_dict

def clip_rank_top_k(model,k,max_rank):
    '''
    Budget constraint clipping
    Returns a Dict() of clipped ranks for each layer as {layer_name: clipped_rank}
    Retains top k rank vectos
    To use this add args `--clip_strategy budget` in the cmd
    '''
    from tqdm import tqdm
    select_count = 0
    rank_dict = {}
    for select_count in tqdm(range(k)):
        rank_norm = {}
        c = 0
        for name, module in model.named_modules():
            if isinstance(module, Linear):
                if name not in rank_dict.keys():
                    rank_dict[name] = 0
                rank = rank_dict[name]
                if rank == max_rank:
                    continue
                last_vector_A = module.lora_A['default'].weight[rank,:]
                last_vector_B = module.lora_B['default'].weight[:,rank]
                normA = torch.norm(last_vector_A, p=2)
                normB = torch.norm(last_vector_B, p=2)
                
                # first norm for normalization 
                last_vector_A = module.lora_A['default'].weight[0,:]
                last_vector_B = module.lora_B['default'].weight[:,0]
                normA0 = torch.norm(last_vector_A, p=2)
                normB0 = torch.norm(last_vector_B, p=2)
                
                rank_norm[name]= min(1,(normA+normB)/(normA0+normB0))
                c+=1
                
        max_name = max(rank_norm.keys(), key=lambda k: (rank_norm[k], -rank_dict[k]))
        rank_dict[max_name] += 1
    
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            module.rank_ = rank_dict[name]
    return rank_dict


def plot_layer(rank_dict,number_layer,exp):
    ''' Plots a heatmap showcasing rank distribution for the LoRA adaptor'''
    ranks = []
    for rank in range(number_layer):
        elem = {}
        for layer in rank_dict.keys():
            if f'.{rank}.' in layer:
                elem[layer.split(f'.{rank}.')[-1]] = rank_dict[layer]
        ranks.append(elem)
    
    target_modules = list(ranks[0].keys())
    data = np.array([np.array([l[m] for m in target_modules]) for l in ranks]).T
    data = pd.DataFrame(data, index = target_modules,
                  columns = [f"layer {i}" for i in range(number_layer)])
    plt.figure(figsize=(number_layer, 5))
    sns.heatmap(data, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
    plt.title("Rank Distribution chart")
    plt.savefig(f"./outputs/{exp}/LORA_.png")

    
def trainable_params(ranks_dict,model):
    ''' Print trainable params post clipping of LoRA adaptor '''
    trainable_params = {}
    for name, module in model.named_modules(): 
       if name in ranks_dict and hasattr(module, 'weight'): 
           shape = module.weight.shape 
           rank = ranks_dict[name]
           param_count = rank * (shape[0] + shape[1])  
    
           trainable_params[name] = param_count

    total_trainable = sum(trainable_params.values()) 
    print(f"Total trainable parameters (NormAL LoRA): {total_trainable}")
             

class ProgressCallback(TrainerCallback):
    '''
    Callback class for handling LoRA clipping and regularization
    
    __init__ args
    reg_end      : Regularization ends on this step
    clip_stage   : LoRA weights are clipped on this step
    name         : Experiment name
    clip_strategy: Which clipping method to use
    clip_mode    : Overides reg_end and clip_stage and switches to thresholding of running expoential average of loss difference
                   Modify the thresholds methods `end_regulariszation` and `clip_lora`
    '''
    def __init__(self,reg_end,clip_stage,name,clip_strategy,clip_mode):
        self.reg_end = reg_end
        self.clip_stage = clip_stage
        self.regularize = True
        self.to_clip = True
        self.name = name
        self.clip_strategy = clip_strategy
        self.clip_mode = clip_mode
        self.running_loss_avg_dif = None
    
    def end_regulariszation(self,state):
        '''Returns when to stop regualrization based on running exponential average'''
        if self.clip_mode == 'manual':
            return state.global_step >= self.reg_end
        elif self.clip_mode == 'auto':
            if self.running_loss_avg_dif is not None and self.running_loss_avg_dif < 8e-2:
                return True
        return False

    def clip_lora(self,state):
        '''Returns when to clip/prune LoRA weights based on running exponential average'''
        if self.clip_mode == 'manual':
            return state.global_step >= self.clip_stage
        elif self.clip_mode == 'auto':
            if self.running_loss_avg_dif is not None and self.running_loss_avg_dif < 50e-3:
               return True
        return False

    def on_log(self,args, state, control,logs=None, **kwargs):
        ''' Logs loss in each step and updates the running exponential average of loss differnce '''
        if 'loss' not in logs.keys():
            return
        loss = logs['loss']
        if self.running_loss_avg_dif == None:
            self.running_loss_avg_dif = loss
            self.prev = loss
        else:
            alpha = 0.9
            self.running_loss_avg_dif = alpha * self.running_loss_avg_dif + (1 - alpha) * abs(loss - self.prev)
            self.prev = loss

    def on_step_end(self, args, state, control, **kwargs):
        ''' Triggers at the end of each step to check for key events like stopping regularization or clipping LoRA '''
        if self.trainer.mode =='no-jinga':
            return # in `no-jinga` mode vanilla LoRA training regime is used so we return 
        if self.regularize:
            if self.end_regulariszation(state):  # check if we can stop regualrization
                model = kwargs["model"]
                self.trainer.jinga_loss = False
                print(f"✅ Reached training step {state.global_step}. END of regularization phase. ")
                self.regularize = False
                self.running_loss_avg_dif = None
        if self.to_clip:
            if self.clip_lora(state): # check if model can be clipped
                model = kwargs["model"]
                with torch.no_grad():
                    if self.clip_strategy == 'budget':
                        ranks_dict = clip_rank_top_k(model, self.trainer.k, self.trainer.max_rank)
                    else:
                        ranks_dict = clip_rank_threshold(model, self.trainer.k)

                    # save clipped ranks in *.json 
                    with open(f'outputs/{self.name}/full_rank.json', 'w') as f:
                        json.dump(ranks_dict, f)
                    mask_clipped(model,self.trainer.reset)
                    display_ranks(model,self.trainer.max_rank)
                print(f"✅ Reached training step {state.global_step}. Model CLIPPED")
                trainable_params(ranks_dict,model)
                self.to_clip = False

class BatchLossLogger(TrainerCallback):
    '''
    Callback class to save step wise loss, disabled by default
    To enable use `--loss yes` in cmd line
    Saves losses.json in `./ouputs/{exp-name}/`
    '''
    def __init__(self):
        self.losses = []

    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history and "loss" in state.log_history[-1]:
            self.losses.append(state.log_history[-1]["loss"])


def image_collate_fn(examples):
    ''' Collate func for Image tasks CIFAR etc '''
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values" : pixel_values, "labels":labels}
