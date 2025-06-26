from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,DebertaV2Tokenizer,BartForConditionalGeneration, BartTokenizer,AutoModelForQuestionAnswering,AutoImageProcessor,ViTForImageClassification,AutoModelForImageClassification
import torch

def build_model(task,model_name,pretrained=None):
    if task == 'MNLI':
        num_class = 3
    elif task == 'COLA' or task == 'QNLI' or task == 'QQP' or task == 'SST' or task == 'RTE' or task == 'MRPC' :
        num_class = 2
    elif task == 'STSB' :
        num_class = 1
    elif task == 'CIFAR':
        num_class = 10
    elif task == 'CIFAR100':
        num_class = 100
    elif task == 'FOOD101':
        num_class = 101
    else:
        num_class = None

    if model_name== 'TinyLlama':
        model_name = 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'
        if pretrained:
            model = AutoModelForCausalLM.from_pretrained(pretrained)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
    elif model_name == 'deberta-base':
        model_name = 'microsoft/deberta-v3-base'
        if task == 'SQUAD':
            model = AutoModelForQuestionAnswering.from_pretrained(model_name,device_map="cuda")
        else:
            if pretrained:
                model = AutoModelForSequenceClassification.from_pretrained(pretrained,num_labels=num_class)
                model.classifier.load_state_dict(torch.load(f"{pretrained}/classifier.pth"))
                model.pooler.load_state_dict(torch.load(f"{pretrained}/pooler.pth"))
            else:
                model=AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=num_class, device_map="cuda")
        tokenizer=AutoTokenizer.from_pretrained(model_name)
        
    elif model_name == 'bart-large':
        model_name = 'facebook/bart-large'
        if pretrained:
            model = BartForConditionalGeneration.from_pretrained(pretrained)
        else:
            model = BartForConditionalGeneration.from_pretrained(model_name,device_map="cuda")
        tokenizer = BartTokenizer.from_pretrained(model_name)
        
    elif model_name == 'vit-base':
        model_name = "google/vit-base-patch16-224"
        if pretrained:
            model = AutoModelForImageClassification.from_pretrained(pretrained)
        else:
            model =  AutoModelForImageClassification.from_pretrained(model_name,num_labels=num_class,device_map='cuda',ignore_mismatched_sizes=True)
        tokenizer = AutoImageProcessor.from_pretrained(model_name)
        
    elif model_name == 'vit-tiny':
        model_name = "WinKawaks/vit-tiny-patch16-224"
        if pretrained:
            model = AutoModelForImageClassification.from_pretrained(pretrained)
        else:
            model = AutoModelForImageClassification.from_pretrained(model_name,num_labels=num_class,device_map="cuda",ignore_mismatched_sizes=True)
        tokenizer = AutoImageProcessor.from_pretrained(model_name)
        
    elif model_name == 'vit-large':
        model_name = "google/vit-large-patch16-224"
        if pretrained:
            model = AutoModelForImageClassification.from_pretrained(pretrained)
        else:
            model =  AutoModelForImageClassification.from_pretrained(model_name,num_labels=num_class,device_map='cuda',ignore_mismatched_sizes=True)
        tokenizer = AutoImageProcessor.from_pretrained(model_name)
        
    else:
        raise Exception("Model not supported. please add in models.model.build_model")
    
    return model, tokenizer

    
