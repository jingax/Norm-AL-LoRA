import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from peft import PeftModel
from datasets import load_dataset
from evaluate import load
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp', help="task to run")
parser.add_argument('--split', type=str, default='validation', help="task to run")
parser.add_argument('--name', help='folder name')
parser.add_argument("--model",type="str",default="None",help="model type")
args = parser.parse_args()

print("Loading model...")
model = AutoModelForQuestionAnswering.from_pretrained(f"./outputs/{args.name}", device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model.eval()

print("Loading dataset...")
datasets = load_dataset("squad_v2")
validation_examples = datasets["validation"]

def preprocess(example):
    tokenized = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_token_type_ids=True,
        padding="max_length"
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["example_id"] = []
    tokenized["answers"] = []

    for i in range(len(tokenized["input_ids"])):
        tokenized["example_id"].append(example["id"][sample_mapping[i]])
        tokenized["answers"].append(example["answers"][sample_mapping[i]])

    return tokenized

tokenized_val = validation_examples.map(preprocess, batched=True, remove_columns=validation_examples.column_names)

val_torch_dataset = tokenized_val.remove_columns(["example_id", "offset_mapping"])
val_torch_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

all_start_logits = []
all_end_logits = []

with torch.no_grad():
    for batch in torch.utils.data.DataLoader(val_torch_dataset, batch_size=16):
        inputs = {k: v.to(model.device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
        outputs = model(**inputs)
        all_start_logits.append(outputs.start_logits.cpu().numpy())
        all_end_logits.append(outputs.end_logits.cpu().numpy())

start_logits = np.concatenate(all_start_logits, axis=0)
end_logits = np.concatenate(all_end_logits, axis=0)

def postprocess_predictions(examples, features, start_logits, end_logits, n_best=20, max_answer_length=30):
    features_per_example = {}
    for i, f in enumerate(features):
        example_id = f["example_id"]
        features_per_example.setdefault(example_id, []).append((i, f))

    predictions = []
    for example in examples:
        example_id = example["id"]
        context = example["context"]
        answers = []

        for i, feature in features_per_example.get(example_id, []):
            start_idx = np.argsort(start_logits[i])[-n_best:]
            end_idx = np.argsort(end_logits[i])[-n_best:]
            offset_mapping = feature["offset_mapping"]

            for start in start_idx:
                for end in end_idx:
                    if start <= end and end - start + 1 <= max_answer_length:
                        if offset_mapping[start] and offset_mapping[end]:
                            start_char = offset_mapping[start][0]
                            end_char = offset_mapping[end][1]
                            answer_text = context[start_char:end_char]
                            score = start_logits[i][start] + end_logits[i][end]
                            answers.append({"text": answer_text, "score": score})

        if answers:
            best_answer = sorted(answers, key=lambda x: x["score"], reverse=True)[0]["text"]
        else:
            best_answer = ""
        predictions.append({"id": example_id, "prediction_text": best_answer, "no_answer_probability": 0.0})

    return predictions

predictions = postprocess_predictions(validation_examples, tokenized_val, start_logits, end_logits)

metric = load("squad_v2")
results = metric.compute(predictions=predictions, references=[{"id": ex["id"], "answers": ex["answers"]} for ex in validation_examples])

print("Outputs saved successfully...")
print("Exact Match:", results["exact"])
print("F1 Score:", results["f1"])
