from transformers import AutoModelForSequenceClassification,AutoTokenizer
from datasets import load_dataset
from evaluate import load
import torch
import os
import csv
import argparse

spearman_metric = load("spearmanr")
pearson_metric = load("pearsonr")

parser = argparse.ArgumentParser()
parser.add_argument('--exp',help="task to run")
parser.add_argument('--name',help='folder name')
parser.add_argument('--split',type=str,default='validation',help="task to run")
parser.add_argument("--model",type="str",default="None",help="model type")
args = parser.parse_args()

print("Loading model...")
model=AutoModelForSequenceClassification.from_pretrained(f"./outputs/{args.name}",device_map="cuda",num_labels=1)
tokenizer=AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
classifier_weights = torch.load(f"./outputs/{args.name}/classifier.pth")
model.classifier.load_state_dict(classifier_weights)
model.pooler.load_state_dict(torch.load(f"./outputs/{args.name}/pooler.pth"))
model.eval()

print("Loading the dataset...")
dataset=load_dataset("glue","stsb")

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_model(model,dataset):
    predictions,labels=[],[]
    results = {}
    for example in dataset[args.split]:
        inputs = tokenizer(example["sentence1"],example["sentence2"],padding="max_length",max_length=128,truncation=True,return_tensors="pt")
        inputs = {k:v.to(device) for k,v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs).logits.cpu().numpy()

        pred = output.squeeze()
        results[example["idx"]] = round(float(pred),3)
        predictions.append(pred)
        labels.append(example["label"])

    # saving the predictions to tsv file
    with open(f'./outputs/{args.name}/STSB.tsv', "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["index", "prediction"])
            for idx in sorted(results.keys()):
                writer.writerow([idx, results[idx]])

    return spearman_metric.compute(predictions=predictions,references=labels),pearson_metric.compute(predictions=predictions,references=labels)

print("Outputs saved successfully...")
spearman,pearson = evaluate_model(model,dataset)
print(f"Spearman score: {spearman['spearmanr']:.4f}")
print(f"Pearson score: {pearson['pearsonr']:.4f}")
