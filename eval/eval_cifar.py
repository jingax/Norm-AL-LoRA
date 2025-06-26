from transformers import ViTForImageClassification, AutoImageProcessor
from datasets import load_dataset
from evaluate import load
import torch
import numpy as np
import os
import argparse
import csv
from PIL import Image

# Load metrics
accuracy_metric = load("accuracy")

parser = argparse.ArgumentParser()
parser.add_argument('--exp', help="task to run")
parser.add_argument('--split', type=str, default='validation', help="task to run")
parser.add_argument('--name', help='folder name')
parser.add_argument('--model',help='model_name')
args = parser.parse_args()

print("Loading model...")
model=ViTForImageClassification.from_pretrained(f"./outputs/{args.name}",num_labels=10,device_map="cuda",ignore_mismatched_sizes=True)
if args.model == 'vit-base':
    model_name = "google/vit-base-patch16-224"
elif args.model == 'vit-large':
    model_name = "google/vit-large-patch16-224"
elif args.model == 'vit-tiny':
    model_name = "WinKawaks/vit-tiny-patch16-224"
processor=AutoImageProcessor.from_pretrained(model_name)

model.classifier.load_state_dict(torch.load(f"./outputs/{args.name}/classifier.pth"))
model.eval()

print("Loading the dataset...")
dataset = load_dataset("uoft-cs/cifar10")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def evaluate_model(model, dataset):
    predictions, labels = [], []
    results = {}
    idx = 0
    for example in dataset[args.split]:
        img = example["img"]
        label = example["label"]
        example["idx"] = idx
        idx = idx+1
        img = img.convert("RGB")

        inputs = processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs).logits.cpu().numpy()

        pred = np.argmax(outputs, axis=-1).item()
        results[example["idx"]] = pred
        predictions.append(pred)
        labels.append(label)
        
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]

    # saving the predictions to tsv file
    with open(f'./outputs/{args.name}/CIFAR10.tsv', "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["index", "prediction"])
        for idx in sorted(results.keys()):
            writer.writerow([idx, results[idx]])

    return {"accuracy": accuracy}

print("Outputs saved successfully...")
res = evaluate_model(model, dataset)
print(f"Accuracy: {res['accuracy']:.4f}")
