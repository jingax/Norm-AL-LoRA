from peft import get_peft_model, LoraConfig, TaskType

def get_lora_config(task,rank,alpha):
    if task == 'NLG':
        target_modules = ["q_proj", "k_proj", "v_proj"] 
        task_type = TaskType.CAUSAL_LM
    elif task == 'SQUAD':
        target_modules = ["query_proj", "value_proj","key_proj",'intermediate.dense','attention.output.dense','output.dense']
        task_type = TaskType.QUESTION_ANS
    elif task == 'XSUM' or task == 'CNN':
        target_modules = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.out_proj","encoder_attn.q_proj","encoder_attn.k_proj","encoder_attn.v_proj","encoder_attn.out_proj","fc1","fc2"]
        task_type = TaskType.SEQ_2_SEQ_LM
    elif task == 'MNLI' or task == 'MRPC' or task == 'COLA' or task == 'SST' or task == 'RTE' or task == 'QQP' or task == 'QNLI' or task == 'STSB':
        target_modules=["query_proj", "value_proj","key_proj",'intermediate.dense','attention.output.dense','output.dense']
        task_type = TaskType.SEQ_CLS
    elif task == 'CIFAR' or 'FOOD101' or 'CIFAR100' :
        target_modules = ["query", "key", "value"]
        # task_type = TaskType.FEATURE_EXTRACTION
        task_type = None
    else:
        raise Exception("Unsupported task") 

    if alpha==0:
        l_a = rank*2
    else:
        l_a = alpha
        
    lora_config = LoraConfig(
        task_type=task_type,  
        target_modules=target_modules, 
        inference_mode=False,  
        r=rank,  
        lora_alpha=l_a,  
        lora_dropout=0.1  
    )
    return lora_config, target_modules
