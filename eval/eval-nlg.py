import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from peft.tuners.lora import LoraLayer,Linear

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',help="task to run")
    parser.add_argument('--name',help='folder name')
    parser.add_argument('--split',help="test/validation")
    parser.add_argument("--model",type="str",default="None",help="model type")
    args = parser.parse_args()

    print("HI")
    # Step 1: Load the pre-trained LLM and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"  # Replace with the model of your choice
    model = AutoModelForCausalLM.from_pretrained(f"./outputs/{args.name}/").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()


    import csv
    data= []
    res = []
    gt = []
    with open('data/e2e-dataset/e2e-dataset/testset_w_refs.csv', mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            x = lines[0] + " <SEP> "
            
            if x not in data:
                data.append(x)
                gt.append('\n')
            gt.append(lines[1]+'\n')

    for input_text in tqdm(data[1:]):
        inputs =  tokenizer(input_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        output = model.generate(input_ids,
            do_sample=False,
            max_length=100,
            attention_mask=attention_mask)
        out = tokenizer.decode(output[0], skip_special_tokens=True,attention_mask=attention_mask)
        res.append(out.split(" <SEP> ")[1].strip().replace('\n','')+'\n')
        
    file1 = open(f'outputs/{args.name}/myfile.txt', 'w')
    file1.writelines(res)
    file1.close()

if __name__ == "__main__":
    main()
