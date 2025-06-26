import torch
from transformers import Trainer
from types import MethodType

def polynomial_rank_penality(self,model,k):
    '''
    Regularize each rank vector with power k, can use other functions as well like math.log or math.exp
    '''
    jinga_loss = 0
    for name, param in model.named_parameters():
        if "lora_A" in name and param.requires_grad:  
            for rank in range(self.max_rank):
                jinga_loss += ((rank+1)**k) * torch.norm(param[rank,:], p=2)
                # jinga_loss += math.log(float(rank+1)) * torch.norm(param[rank,:], p=2  # use this for math.log penality
                
        elif "lora_B" in name and param.requires_grad:  
            for rank in range(self.max_rank):
                jinga_loss += ((rank+1)**k) * torch.norm(param[:,rank], p=2)
                # jinga_loss += math.log(float(rank+1)) * torch.norm(param[:,rank], p=2) # use this for math.log penality

    return jinga_loss/self.max_rank

def margin_rank_penality(self,model):
    '''
    This loss enforces a margin between each succesive rank vector by penalizing if norm of a rank vecotr
    is greater than norm of previous rank vector
    '''
    jinga_loss = 0  
    for name, param in model.named_parameters():
        if "lora_A" in name and param.requires_grad:  
            norms = torch.norm(param,dim=1)
        elif "lora_B" in name and param.requires_grad:
            norms = torch.norm(param,dim=0)
        else:
            continue
        penality = norms[1:] - norms[:-1]
        margin = torch.tensor([0.05/(r+1) for r in range(self.max_rank - 1)],device=penality.device)
        penality += margin
        jinga_loss += sum(torch.relu(penality))
                
    return jinga_loss   

# unbound method compute_loss from Trainer class
orig_compute_loss = Trainer.compute_loss

def regularisation_injected_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    '''
    Monkey patching the compute_loss to add regularization.
    If you wish to use margin_rank_penality for regularization modify in this method accordingly
    Note: This patching is based on `transformers==4.48.0`, if you are using a differnt version of tranformers
    then verify that method signature for compute_loss is same OR modify it accordingly.
    '''
    if return_outputs:
        loss, outputs = orig_compute_loss(self,model, inputs, return_outputs, num_items_in_batch)
    else:
        loss = orig_compute_loss(self,model, inputs, return_outputs, num_items_in_batch)
    if self.mode == 'jinga':
        jinga_loss = 0
        if self.jinga_loss:
            jinga_loss = self.regularization_weight * self.polynomial_rank_penality(model,self.p)
            # jinga_loss = self.regularization_weight * self.margin_rank_penality(model)    
        
        loss = loss + jinga_loss
    return (loss, outputs) if return_outputs else loss


def inject_regularization(trainer,args):
    '''
    We monkey patch the compute_loss function in the Trainer class to add regularization
    and add other required attributes to the Trainer object
    '''
    trainer.regularization_weight = args.reg_w
    trainer.max_rank = args.rank
    trainer.jinga_loss = True
    trainer.to_clip = True
    trainer.reg_start = args.reg_start
    trainer.reg_end = args.reg_end
    trainer.clip_stage = args.clip_stage
    trainer.k = args.k
    trainer.reset = args.reset
    trainer.p = args.p
    trainer.mode = args.mode
    
    trainer.polynomial_rank_penality = MethodType(polynomial_rank_penality, trainer)
    trainer.margin_rank_penality = MethodType(margin_rank_penality, trainer)
    trainer.compute_loss = MethodType(regularisation_injected_compute_loss, trainer)
    
    return trainer
    
