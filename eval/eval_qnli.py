from transformers import AutoModelForSequenceClassification,AutoTokenizer
from datasets import load_dataset
from evaluate import load
import torch
import numpy as np
import os
import argparse
import csv

metric=load("accuracy")

parser = argparse.ArgumentParser()
parser.add_argument('--exp',help="experiment name")
parser.add_argument('--name',help='folder name')
parser.add_argument('--split',type=str,default='validation',help="task to run")
parser.add_argument("--model",type="str",default="None",help="model type")
args = parser.parse_args()

print("Loading the model...")
model=AutoModelForSequenceClassification.from_pretrained(f"./outputs/{args.name}",device_map="cuda")
tokenizer=AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model.classifier.load_state_dict(torch.load(f"./outputs/{args.name}/classifier.pth"))
model.pooler.load_state_dict(torch.load(f"./outputs/{args.name}/pooler.pth"))
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading the dataset...")
dataset=load_dataset("glue","qnli")

def evaluate_model(model,dataset):
    predictions,labels=[],[]
    results = {}
    for example in dataset[args.split]:
        inputs = tokenizer(example['question'],example['sentence'],padding="max_length",max_length=128,truncation=True,return_tensors="pt")
        inputs = {k: v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            output = model(**inputs).logits.cpu().numpy()

        pred = int(np.argmax(output,axis=-1).item())
        results[example["idx"]] = pred
        predictions.append(pred)
        labels.append(example['label'])

        # saving the predictions to tsv file
        with open(f'./outputs/{args.name}/QNLI.tsv', "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["index", "prediction"])
            for idx in sorted(results.keys()):
                writer.writerow([idx, "not_entailment" if results[idx]==0 else "entailment" ])
                
    return metric.compute(predictions=predictions,references=labels)
        
print("Outputs successfully saved...")
res = evaluate_model(model,dataset)
print(f"Accuracy: {res['accuracy']:.4f}")
