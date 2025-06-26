from transformers import AutoModelForSequenceClassification,AutoTokenizer
from datasets import load_dataset
from evaluate import load
import torch
import numpy as np
import os
import argparse
import csv

accuracy_metric=load("accuracy")
f1_metric=load("f1")

parser = argparse.ArgumentParser()
parser.add_argument('--exp',help="task to run")
parser.add_argument('--split',type=str,default='validation',help="task to run")
parser.add_argument('--name',help='folder name')
parser.add_argument("--model",type="str",default="None",help="model type")
args = parser.parse_args()

print("Loading model...")
model=AutoModelForSequenceClassification.from_pretrained(f"./outputs/{args.name}",device_map="cuda")
tokenizer=AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

model.classifier.load_state_dict(torch.load(f"./outputs/{args.name}/classifier.pth"))
model.pooler.load_state_dict(torch.load(f"./outputs/{args.name}/pooler.pth"))
model.eval()

print("Loading the dataset...")
dataset=load_dataset("glue","mrpc")

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_model(model,dataset):
    sentences1,sentences2,predictions,labels=[],[],[],[]
    results = {}
    for example in dataset[args.split]:
        sent1 = example["sentence1"]
        sent2 = example["sentence2"]
        sentences1.append(sent1)
        sentences2.append(sent2)
        inputs = tokenizer(sent1,sent2,padding="max_length",max_length=128,truncation=True,return_tensors="pt")
        inputs = {k: v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            output = model(**inputs).logits.cpu().numpy()

        pred = int(np.argmax(output,axis=-1).item())
        results[example["idx"]] = pred
        predictions.append(pred)
        labels.append(example['label'])

        # saving the predictions to tsv file
        with open(f'./outputs/{args.name}/MRPC.tsv', "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["index", "prediction"])
            for idx in sorted(results.keys()):
                writer.writerow([idx, results[idx]]))
             
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions,references=labels)["accuracy"],
        "f1":f1_metric.compute(predictions=predictions,references=labels)["f1"],
    }
        
print("Outputs successfully saved...")
res = evaluate_model(model,dataset)
print(f"Accuracy: {res['accuracy']:.4f}")
print(f"F1 : {res['f1']:.4f}")
