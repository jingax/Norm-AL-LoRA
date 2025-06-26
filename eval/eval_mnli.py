from evaluate import load
from transformers import DebertaV2Tokenizer,AutoModelForSequenceClassification,Trainer,AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.functional import softmax
import argparse

accuracy_metric=load("accuracy")
f1_metric=load("f1")

print("Loading fine-tuned model...")
parser = argparse.ArgumentParser()
parser.add_argument('--exp',help="task to run")
parser.add_argument('--name',help='folder name')
parser.add_argument('--split',help="dataset to run")
parser.add_argument("--model",type="str",default="None",help="model type")
args = parser.parse_args()
model=AutoModelForSequenceClassification.from_pretrained(f"./outputs/{args.name}",device_map="cuda",num_labels=3)
tokenizer=AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

model.classifier.load_state_dict(torch.load(f"./outputs/{args.name}/classifier.pth"))
model.pooler.load_state_dict(torch.load(f"./outputs/{args.name}/pooler.pth"))
model.eval()

print("Evaluating on test")

test_matched=load_dataset("glue","mnli",split="validation_matched")
test_mismatched=load_dataset("glue","mnli",split="validation_mismatched")

mappping = test_matched.features['label'].names

def tokenize_function(examples):
    x = tokenizer(examples['premise'],examples['hypothesis'],padding="max_length",max_length=512,truncation=True)
    x['idx'] = examples['idx']
    return x

tokenized_test_matched=test_matched.map(tokenize_function,batched=True)
tokenized_test_mismatched=test_mismatched.map(tokenize_function,batched=True)

batch_size=32

def func(batch):
    input_ids=torch.tensor([item["input_ids"] for item in batch])
    attention_mask=torch.tensor([item["attention_mask"] for item in batch])
    label=torch.tensor([item["label"] for item in batch])
    idx = torch.tensor([item["idx"] for item in batch])
    return {"input_ids":input_ids,"attention_mask":attention_mask,"label":label,"idx":idx}

matched_data=DataLoader(tokenized_test_matched,batch_size=batch_size,collate_fn=func)
mismatched_data=DataLoader(tokenized_test_mismatched,batch_size=batch_size,collate_fn=func)

def evaluate_model(dataloader):
    predictions, references ,idx= [], [],[]
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits
            preds=torch.argmax(logits,dim=1).cpu().numpy()
            labels = batch["label"].cpu().numpy()
            idx_ = batch["idx"].cpu().numpy()
            
        predictions.extend(preds)
        references.extend(labels)
        idx.extend(idx_)

    acc = accuracy_metric.compute(predictions=predictions, references=references)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=references, average="macro")["f1"]
      
    return acc, f1

matched_acc,f1_matched=evaluate_model(matched_data)
mismatched_acc,f1_mismatched=evaluate_model(mismatched_data)

print(f"matched accuracy: {matched_acc:.5f}")
print(f"mismatched accuracy: {mismatched_acc:.5f}")
print(f"matched f1: {f1_matched:.5f}")
print(f"mismatched f1: {f1_mismatched:.5f}")
