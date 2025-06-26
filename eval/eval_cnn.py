from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset
import evaluate
import torch
import numpy as np
import os
import argparse
import csv

metric=evaluate.load("rouge")

parser = argparse.ArgumentParser()
parser.add_argument('--exp',help="task to run")
parser.add_argument('--name',help='folder name')
parser.add_argument('--split',type=str,default='validation',help="task to run")
parser.add_argument("--model",type="str",default="None",help="model type")
args = parser.parse_args()

print("Loading model...")
model=BartForConditionalGeneration.from_pretrained(f"./outputs/{args.name}",device_map="cuda")
tokenizer=BartTokenizer.from_pretrained("facebook/bart-large")
model.eval()

print("Loading the dataset...")
dataset=load_dataset("cnn_dailymail", "3.0.0")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def evaluate_model(model,dataset,split):
        predictions,original = [],[]
        results = {}
        i=0
        for example in dataset[split]:
            inputs = tokenizer(example["article"],padding="max_length",max_length=1024,truncation=True,return_tensors="pt")
            inputs = {k: v.to(device) for k,v in inputs.items()}
            with torch.no_grad():
               output = model.generate(**inputs,max_length=256,num_beams=4,early_stopping=True)
    
            pred = tokenizer.decode(output[0],skip_special_tokens=True)
            results[example['id']] = pred
            predictions.append(pred)
            original.append(example["highlights"])
            
        # saving the predictions to tsv file
        with open(f'./outputs/{args.name}/CNN.tsv', "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["index", "prediction"])
            for idx in sorted(results.keys()):
                writer.writerow([idx, results[idx]])
                
        return metric.compute(predictions=predictions,references=original)

print("Outputs successfully saved...")
res=evaluate_model(model,dataset,args.split)
print(f"Rouge-1: {res['rouge1']:.4f}")
print(f"Rouge-2: {res['rouge2']:.4f}")
print(f"Rouge-3: {res['rougeL']:.4f}")
