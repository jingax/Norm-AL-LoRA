from datasets import Dataset
from datasets import load_dataset
import json
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
from PIL import Image
from collections import defaultdict

def NGL_task(tokenizer):
    '''
    Returns dataset for E2E Natural language gneration challenge
    '''
    import csv
    data= []
    # download e2e dataset from `https://github.com/tuetschek/e2e-dataset.git` 
    with open('data/e2e-dataset/e2e-dataset/trainset.csv', mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            data.append({"input_text": lines[0] +" <SEP> "+lines[1], "output_text": lines[0] +" ? "+lines[1]})
    data = data[1:]  # remove header
    def preprocess(example):
        inputs = tokenizer(example["input_text"], truncation=True, padding="max_length", max_length=250)
        inputs["labels"]=inputs["input_ids"]
        return inputs

    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(preprocess)

    columns_to_keep = ["input_ids","attention_mask","labels"]
    tokenized_dataset.set_format(type="torch",columns=columns_to_keep)

    return tokenized_dataset, None

def MNLI(tokenizer,split='train'):
    dataset = load_dataset("glue", "mnli")

    def tokenize_function(examples):
        padding = "max_length"
        if 'test' in split:
            padding = False
        model_inputs= tokenizer(examples["premise"],examples["hypothesis"],padding=padding,max_length=512,truncation=True)
        model_inputs["labels"]=examples["label"]
        return model_inputs

    tokenized_dataset = dataset[split].map(tokenize_function, batched=False,num_proc=1)
    
    columns_to_keep = ["input_ids", "attention_mask", "label"]
    tokenized_dataset.set_format(type="torch", columns=columns_to_keep)
    
    return tokenized_dataset, None

def STSB(tokenizer,split=None):
    dataset = load_dataset("glue","stsb")

    def tokenize_function(examples):
         model_inputs = tokenizer(examples["sentence1"],examples["sentence2"],padding="max_length",max_length=128,truncation=True)
         return model_inputs

    tokenized_train_dataset = dataset["train"].map(tokenize_function,batched=True)
    tokenized_eval_dataset = dataset["validation"].map(tokenize_function,batched=True)

    columns_to_keep = ["input_ids","attention_mask","label"]
    tokenized_train_dataset.set_format(type="torch",columns=columns_to_keep)
    tokenized_eval_dataset.set_format(type="torch",columns=columns_to_keep)

    return tokenized_train_dataset,tokenized_eval_dataset

def COLA(tokenizer,split=None):
    dataset = load_dataset("glue","cola")

    def tokenize_function(examples):
        return tokenizer(examples["sentence"],padding="max_length",max_length=128,truncation=True)

    tokenized_train_dataset=dataset["train"].map(tokenize_function,batched=True)
    tokenized_eval_dataset=dataset["validation"].map(tokenize_function,batched=True)

    columns_to_keep = ["input_ids","attention_mask","label"]
    tokenized_train_dataset.set_format(type="torch",columns=columns_to_keep)
    tokenized_eval_dataset.set_format(type="torch",columns=columns_to_keep)

    return tokenized_train_dataset,tokenized_eval_dataset

def QNLI(tokenizer,split=None):
    dataset = load_dataset("glue","qnli")

    def tokenize_function(examples):
        return tokenizer(examples["question"],examples["sentence"],padding="max_length",max_length=128,truncation=True)

    tokenized_train_dataset = dataset["train"].map(tokenize_function,batched=True)
    tokenized_eval_dataset = dataset["validation"].map(tokenize_function,batched=True)

    columns_to_keep = ["input_ids","attention_mask","label"]
    tokenized_train_dataset.set_format(type="torch",columns=columns_to_keep)
    tokenized_eval_dataset.set_format(type="torch",columns=columns_to_keep)

    return tokenized_train_dataset,tokenized_eval_dataset

def QQP(tokenizer,split=None):
    dataset = load_dataset("glue","qqp")

    def tokenize_function(examples):
        return tokenizer(examples["question1"],examples["question2"],padding="max_length",max_length=128,truncation=True)

    tokenized_train_dataset = dataset["train"].map(tokenize_function,batched=True)
    tokenized_eval_dataset = dataset["validation"].map(tokenize_function,batched=True)

    columns_to_keep = ["input_ids","attention_mask","label"]
    tokenized_train_dataset.set_format(type="torch",columns=columns_to_keep)
    tokenized_eval_dataset.set_format(type="torch",columns=columns_to_keep)

    return tokenized_train_dataset,tokenized_eval_dataset  

def SST(tokenizer,split=None):
    dataset = load_dataset("glue","sst2")

    def tokenize_function(examples):
        return tokenizer(examples["sentence"],padding="max_length",max_length=128,truncation=True)

    tokenized_train_dataset = dataset["train"].map(tokenize_function,batched=True)
    tokenized_eval_dataset = dataset["validation"].map(tokenize_function,batched=True)

    columns_to_keep = ["input_ids","attention_mask","label"]
    tokenized_train_dataset.set_format(type="torch",columns=columns_to_keep)
    tokenized_eval_dataset.set_format(type="torch",columns=columns_to_keep)

    return tokenized_train_dataset,tokenized_eval_dataset

def RTE(tokenizer,split=None):
    dataset = load_dataset("glue","rte")

    def tokenize_function(examples):
        return tokenizer(examples["sentence1"],examples["sentence2"],padding="max_length",max_length=128,truncation=True)

    tokenized_train_dataset = dataset["train"].map(tokenize_function,batched=True)
    tokenized_eval_dataset = dataset["validation"].map(tokenize_function,batched=True)

    columns_to_keep = ["input_ids","attention_mask","label"]
    tokenized_train_dataset.set_format(type="torch",columns=columns_to_keep)
    tokenized_eval_dataset.set_format(type="torch",columns=columns_to_keep)

    return tokenized_train_dataset,tokenized_eval_dataset

def MRPC(tokenizer,split=None):
    dataset = load_dataset("glue","mrpc")

    def tokenize_function(examples):
       return tokenizer(examples["sentence1"],examples["sentence2"],padding="max_length",max_length=128,truncation=True)

    tokenized_train_dataset = dataset["train"].map(tokenize_function,batched=True)
    tokenized_eval_dataset = dataset["validation"].map(tokenize_function,batched=True)

    columns_to_keep = ["input_ids","attention_mask","label"]
    tokenized_train_dataset.set_format(type="torch",columns=columns_to_keep)
    tokenized_eval_dataset.set_format(type="torch",columns=columns_to_keep)

    return tokenized_train_dataset,tokenized_eval_dataset

def XSUM(tokenizer,split=None):
    dataset = load_dataset("xsum",trust_remote_code=True)

    def tokenize_function(examples):
        inputs = tokenizer(examples["document"],padding="max_length",max_length=512,truncation=True)
        outputs = tokenizer(examples["summary"],padding="max_length",max_length=64,truncation=True)
        inputs['labels']=outputs['input_ids']
        return inputs

    tokenized_train_dataset = dataset["train"].map(tokenize_function,batched=True)
    tokenized_eval_dataset = dataset["validation"].map(tokenize_function,batched=True)
    
    columns_to_keep = ["input_ids","attention_mask","labels"]
    tokenized_train_dataset.set_format(type="torch",columns=columns_to_keep)
    tokenized_eval_dataset.set_format(type="torch",columns=columns_to_keep)

    return tokenized_train_dataset,tokenized_eval_dataset

def SQUAD(tokenizer,split=None):
    dataset = load_dataset("squad_v2")

    def tokenize_function(examples):
       inputs = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            padding="max_length"
       )

       sample_mapping = inputs.pop("overflow_to_sample_mapping")
       inputs["example_id"] = []
       inputs["answers"] = []
       inputs["sequence_ids"]=[]
    
       for i in range(len(inputs["input_ids"])):
            inputs["example_id"].append(examples["id"][sample_mapping[i]])
            inputs["answers"].append(examples["answers"][sample_mapping[i]])
            input_ids = inputs['input_ids'][i]
            sequence_ids = input_ids.copy()
            sequence_ids[0]=None
    
            start_index = 1
            while input_ids[start_index]!=2:
                sequence_ids[start_index]=0
                start_index=start_index+1
    
            sequence_ids[start_index]=None
            start_index=start_index+1
            while input_ids[start_index]!=2:
                sequence_ids[start_index]=1
                start_index=start_index+1

            while start_index<len(input_ids):
                sequence_ids[start_index]=None
                start_index=start_index+1
            inputs['sequence_ids'].append(sequence_ids)
    
       return inputs

    tokenized_train_dataset = dataset['train'].map(tokenize_function,batched=True,remove_columns=dataset['train'].column_names)
    tokenized_eval_dataset = dataset['validation'].map(tokenize_function,batched=True,remove_columns=dataset['validation'].column_names)

    def add_start_end_positions(examples):
            start_positions = []
            end_positions = []
        
            for i in range(len(examples["input_ids"])):
                offsets = examples["offset_mapping"][i]
                input_ids = examples["input_ids"][i]
                answer = examples["answers"][i]
        
                cls_index = input_ids.index(tokenizer.cls_token_id)
        
                if len(answer["answer_start"]) == 0:
                    start_positions.append(cls_index)
                    end_positions.append(cls_index)
                    continue
        
                start_char = answer["answer_start"][0]
                end_char = start_char + len(answer["text"][0])
        
                sequence_ids = examples['sequence_ids'][i]
                token_start_index = 0
                while sequence_ids[token_start_index]!=1:
                    token_start_index += 1
        
                token_end_index = len(offsets) - 1
                while sequence_ids[token_end_index]!=1:
                    token_end_index -= 1
        
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    start_positions.append(cls_index)
                    end_positions.append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    start_positions.append(token_start_index - 1)
        
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_positions.append(token_end_index + 1)
        
            examples["start_positions"] = start_positions
            examples["end_positions"] = end_positions
            return examples

    tokenized_train_dataset = tokenized_train_dataset.map(add_start_end_positions, batched=True)
    tokenized_eval_dataset = tokenized_eval_dataset.map(add_start_end_positions, batched=True)
    
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(["example_id", "answers","offset_mapping",'sequence_ids'])
    tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(["example_id", "answers","offset_mapping",'sequence_ids'])
    
    
    tokenized_train_dataset.set_format("torch")
    tokenized_eval_dataset.set_format("torch")

    return tokenized_train_dataset,tokenized_eval_dataset

def CNN(tokenizer,split=None):
    dataset = load_dataset("cnn_dailymail", "3.0.0",trust_remote_code=True)

    def tokenize_function(examples):
        inputs = tokenizer(examples["article"],padding="max_length",max_length=512,truncation=True)
        outputs = tokenizer(examples["highlights"],padding="max_length",max_length=64,truncation=True)
        inputs['labels']=outputs['input_ids']
        return inputs

    tokenized_train_dataset = dataset["train"].map(tokenize_function,batched=True)
    tokenized_eval_dataset = dataset["validation"].map(tokenize_function,batched=True)
    
    columns_to_keep = ["input_ids","attention_mask","labels"]
    tokenized_train_dataset.set_format(type="torch",columns=columns_to_keep)
    tokenized_eval_dataset.set_format(type="torch",columns=columns_to_keep)

    return tokenized_train_dataset,tokenized_eval_dataset


def CIFAR(processor,split=None):
     dataset = load_dataset("uoft-cs/cifar10")
    
     label_names = list(set(example['label'] for example in dataset['train']))
     print(label_names)
     print(len(label_names))
     label_names.sort()
     print(label_names)
     label2id = {name:i for i,name in enumerate(label_names)}
        
     def preprocess(batch):
        batch["img"] = [image.convert("RGB") for image in batch["img"]]
        inputs = processor(images=batch["img"], return_tensors="pt")
        batch["label"] = [label2id[label] for label in batch["label"]]
        return {
            "pixel_values": inputs["pixel_values"],
            "labels": torch.tensor(batch["label"]),
        }
         
     dataset = dataset.map(preprocess,batched=True)
     dataset.remove_columns(["img"])

     processed_train_dataset = dataset["train"]
     processed_test_dataset = dataset["test"]

     columns_to_keep = ["pixel_values","labels"]
     processed_train_dataset.set_format(type="torch",columns=columns_to_keep)
     processed_test_dataset.set_format(type="torch",columns=columns_to_keep)

     return processed_train_dataset,processed_test_dataset

def CIFAR100(processor,split=None):
     dataset = load_dataset("uoft-cs/cifar100")
     print(dataset)
     label_names = list(set(example['fine_label'] for example in dataset['train']))
     label_names.sort()
     label2id = {name:i for i,name in enumerate(label_names)}
        
     def preprocess(batch):
        batch["img"] = [image.convert("RGB") for image in batch["img"]]
        inputs = processor(images=batch["img"], return_tensors="pt")
        batch["fine_label"] = [label2id[label] for label in batch["fine_label"]]
        return {
            "pixel_values": inputs["pixel_values"],
            "labels": torch.tensor(batch["fine_label"]),
        }
         
     dataset = dataset.map(preprocess,batched=True)
     print(dataset)
     dataset.remove_columns(["img"])

     processed_train_dataset = dataset["train"]
     processed_test_dataset = dataset["test"]

     columns_to_keep = ["pixel_values","labels"]
     processed_train_dataset.set_format(type="torch",columns=columns_to_keep)
     processed_test_dataset.set_format(type="torch",columns=columns_to_keep)

     return processed_train_dataset,processed_test_dataset

def FOOD(processor,split=None):
     dataset = load_dataset("ethz/food101")
         
     label_names = list(set(example['label'] for example in dataset['train']))
     label_names.sort()
     label2id = {name:i for i,name in enumerate(label_names)}
    
     label_to_examples = defaultdict(list)
     for example in dataset["train"]:
         label = example['label']
         if len(label_to_examples[label])<300:
             label_to_examples[label].append(example)

     filtered_examples = []
     for label,example in label_to_examples.items():
         filtered_examples.extend(example)

     dataset["train"] = Dataset.from_list(filtered_examples)
        
     def preprocess(batch):
        batch["image"] = [image.convert("RGB") for image in batch["image"]]
        inputs = processor(images=batch["image"], return_tensors="pt")
        batch["label"] = [label2id[label] for label in batch["label"]]
        return {
            "pixel_values": inputs["pixel_values"],
            "labels": torch.tensor(batch["label"]),
        }

     train_dataset = dataset["train"]
     validation_dataset = dataset["validation"]
     processed_train_dataset = train_dataset.map(preprocess,batched=True)
     processed_validation_dataset = validation_dataset.map(preprocess,batched=True)
     dataset.remove_columns(["image"])
    
     columns_to_keep = ["pixel_values","labels"]
     processed_train_dataset.set_format(type="torch",columns=columns_to_keep)
     processed_validation_dataset.set_format(type="torch",columns=columns_to_keep)
    
     return processed_train_dataset,processed_validation_dataset

def DatasetPack(task,tokenizer,split='train'):
    if task == "NLG":
        return NGL_task(tokenizer)
    elif task == "MNLI":
        return MNLI(tokenizer,split)
    elif task == 'STSB':
        return STSB(tokenizer,split)
    elif task == 'COLA':
        return COLA(tokenizer,split)
    elif task == "QNLI":
        return QNLI(tokenizer,split)
    elif task == "QQP":
        return QQP(tokenizer,split)
    elif task == "SST":
        return SST(tokenizer,split)
    elif task == "RTE":
        return RTE(tokenizer,split)
    elif task == "MRPC":
        return MRPC(tokenizer,split)
    elif task == "XSUM":
        return XSUM(tokenizer,split)
    elif task == "CNN":
        return CNN(tokenizer,split)
    elif task == "SQUAD":
        return SQUAD(tokenizer,split)
    elif task =='CIFAR':
        return CIFAR(tokenizer,split)
    elif task == 'CIFAR100':
        return CIFAR100(tokenizer,split)
    elif task == 'FOOD101' :
        return FOOD(tokenizer,split)
    else:
        raise Exception("Invalid task") 
