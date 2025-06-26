import torch
from peft.tuners.lora import Linear
from types import MethodType
from typing import Any, Optional, Union

def use_mask(self,adapter_name,x):
    '''
    Applies mask over weights and gradients. full/max rank is self.r, and updated/clipped rank is self.rank_ 
    '''
    mask_A_full = torch.zeros_like(self.lora_A[adapter_name].weight)
    mask_A_full[:self.rank_,:] = 1

    mask_B_full = torch.zeros_like(self.lora_B[adapter_name].weight)
    mask_B_full[:,:self.rank_] = 1

    # when weights are clipped (self.rank_ is modidifed) the graidents are also masked this is done only once
    # self.hooked ensures that we do it only once
    if self.lora_A[adapter_name].weight.requires_grad and self.rank_ != self.r[adapter_name] and not self.hooked:
        self.lora_A[adapter_name].weight.register_hook(lambda grad, m=mask_A_full: grad * m)
        self.lora_B[adapter_name].weight.register_hook(lambda grad, m=mask_B_full: grad * m)
        self.hooked = True
    
    mask_A = self.lora_A[adapter_name].weight * mask_A_full
    mask_B = self.lora_B[adapter_name].weight * mask_B_full
    x = torch.matmul(x,mask_A.t())#+self.lora_A[adapter_name].bias
    x = torch.matmul(x,mask_B.t())#+self.lora_B[adapter_name].bias
    if self.lora_B[adapter_name].bias:
        x += self.lora_B[adapter_name].bias
    return x
    
def update_scaling(self,adapter_name):
    ''' Update the scaling factor post clipping of LoRA '''
    self.scaling[adapter_name] =  self.scaling[adapter_name] *  self.r[adapter_name] / self.rank_
    
def apply_mask(self,adapter_name):
    '''
    This method zeros out all weights that have been clipped.
    The attribute rank_ (new clipped rank) is not stored while saving adaptor and zeroing out the clipped weights ensures the saved adaptor behaves as a clipped one
    '''
    mask = torch.zeros_like(self.lora_A[adapter_name].weight)
    mask[:self.rank_,:] = 1
    self.lora_A[adapter_name].weight *= mask  # apply mask 
    mask = torch.zeros_like(self.lora_B[adapter_name].weight)
    mask[:,:self.rank_] = 1
    self.lora_B[adapter_name].weight *= mask  # apply mask 


def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    '''
    This code is adapted from peft commit 4c82bfff76782fe62d7132373bbe2b2974068e7d
    If you are using a differnt version of peft please modify this method accordingly
    only calling of method use_mask needs to be changed in updated code
    '''
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        res = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        res = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
    elif self.merged:
        res = self.base_layer(x, *args, **kwargs)
    else:
        # our LoRA config runs only this else condition. 
        # if you code runs a differnt condition please add masking functionality there too
        res = self.base_layer(x, *args, **kwargs)
        torch_res_dtype = res.dtype

        lora_A_keys = self.lora_A.keys()
        for active_adapter in self.active_adapters:
            # assuming single default adaptor only
            # if you add more adaptors ensure attributes like self.rank_ are also added for each adaptor
            # CURRENTLY global rank_attributes is used

            if active_adapter not in lora_A_keys:
                continue #never true for current implementation

            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter] #alpha
            x = self._cast_input_dtype(x, self.lora_A[active_adapter].weight.dtype)
            if active_adapter not in self.lora_variant:  # vanilla LoRA
                # res = res + lora_B(lora_A(dropout(x))) * scaling
                res = res + self.use_mask(active_adapter,dropout(x)) * scaling
            else:
                res = self.lora_variant[active_adapter].forward(
                    self,
                    active_adapter=active_adapter,
                    x=x,
                    res=res,
                )

        res = res.to(torch_res_dtype)
    return res

def normal_lora_mask_injection(model):
    '''
    Inject masking code to all linear layers of a model
    If you are using any LoRA layer (Embedding or Conv) please inject these attributes to use them with NormAL LoRA
    '''
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            module.rank_ = module.r[list(module.r.keys())[0]]
            module.hooked = False
            module.use_mask = MethodType(use_mask, module)
            module.update_scaling = MethodType(update_scaling, module)
            module.apply_mask = MethodType(apply_mask, module)
            module.forward = MethodType(forward, module)
    return model
            
