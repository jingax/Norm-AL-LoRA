from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, DataCollatorWithPadding
from src.rank_norm_regularisation import inject_regularization
from src.normAL_LoRA import normal_lora_mask_injection
from src.callbacks import ProgressCallback, BatchLossLogger ,image_collate_fn
from datasets import Dataset
from huggingface_hub import login
from peft import get_peft_model, LoraConfig
import json
import argparse
from utils.display import print_trainable_parameters, plot_layer_ranks
from utils.wandb import initialize_wandb
from data.dataset import DatasetPack
from models.models import build_model
from models.lora_config import get_lora_config
from peft.tuners.lora import LoraLayer,Linear
import torch
import yaml
from pathlib import Path
import wandb

def main():
    parser = argparse.ArgumentParser(description="Evaluate code generation predictions.")
    parser.add_argument("--name", type=str, default="test-8", help="dir name to save in")
    parser.add_argument("--task", type=str, default="NLG", help="Task/dataset")
    parser.add_argument("--model_name", type=str, default="TinyLlama", help="Base model name") 
    parser.add_argument("--rank", type=int, default=8, help="Maximum rank or just rank for standard LoRA")
    parser.add_argument("--mode", type=str, default="no-jinga", help="Use NormAL Lora (jinga) or standard LoRA (no-jinga). \
                                                                      Using --mode no-jinga will ignore all args pertaining to NormAL LoRA (like reg_w,clip_mode etc.)")
    parser.add_argument("--reg_start", type=int, default=0, help="Jinga start ")
    parser.add_argument("--reg_end", type=int, default=100, help="jinga end")
    parser.add_argument("--clip_stage", type=int, default=200, help="when clip")
    parser.add_argument("--reg_w", type=float, default=0.001, help="reg weight")
    parser.add_argument("--k", type=int, default=150, help="top_k for clip_strategy budget. If you use zeros as clip_strategy ensure 0 < k < 100")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--reset", type=str, default="no", help="POST clip reset (yes/no)")
    parser.add_argument("--epochs",type=float,default=5.0,help="set epochs according to task")
    parser.add_argument("--lr",type=float,default=0.0001,help="learning rate")
    parser.add_argument("--alpha",type=int,default=0,help="Setting to zero makes it alpha as 2 * rank ")
    parser.add_argument("--loss",type=str,default="no",help="want to store losses or not")
    parser.add_argument("--batch_s",type=int,default=64,help="batch size")
    parser.add_argument("--p", type=float, default=0.5, help="penalizing power, to use other function like math.log or math.exp see \
                                                              method src.rank_norm_regularisation.regularisation_injected_compute_loss")
    parser.add_argument("--clip_strategy",type=str,default="budget",help="clipping strategy budget/zeros")
    parser.add_argument("--clip_mode",type=str,default="manual",help="when clip? manual/auto")
    parser.add_argument("--weight_decay",type=float,default="0.01",help="weight decay")
    parser.add_argument("--report_to",type="str",default="tensorboard",help="write the tool name you want to report the logs to")

    args = parser.parse_args()
    
    # save training args 
    out_path = Path(f'./outputs/{args.name}')
    out_path.mkdir(parents=True,exist_ok=True)
    with open(f'./outputs/{args.name}/config.yaml', 'w') as f:
        yaml.dump(vars(args), f)
    
    # Login to Hugging Face
    hg_token = 'xxxxxxxxxxxxxxxxxxxxxxxxxx'  # Replace with your actual token
    login(hg_token)
    torch.manual_seed(args.seed)

    print("Loading Model and tokenizer...")
    model, tokenizer = build_model(args.task,args.model_name)
    
    # Load data
    print("Loading dataset...")
    train_dataset, eval_dataset = DatasetPack(task=args.task,tokenizer=tokenizer)

    # Define LoRA Configuration
    print("Configuring LoRA...")
    lora_config, target_modules = get_lora_config(args.task,args.rank,args.alpha)

    # Apply LoRA
    print("Applying LoRA to the model...")
    model = get_peft_model(model, lora_config)

    print("Mask injection to enable NormAL LoRA...")
    model = normal_lora_mask_injection(model)
    

    # Print Parameters
    print("Calculating trainable parameters...")
    print_trainable_parameters(model)
 
    if args.loss=='yes':
       logging_s=1
    else:
       logging_s=100

    if args.task == 'CIFAR' or args.task == 'FOOD101' or args.task == 'CIFAR100' :
        datacollator = image_collate_fn
    else:
        datacollator = None

    # wandb initialization
    if args.report_to == "wandb":
        initialize_wandb(args)
        wandb.watch(model,log="all",log_freq=100)
    
    # Training Arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=f"./outputs/{args.name}",
        per_device_train_batch_size=args.batch_s,
        per_device_eval_batch_size=32,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=logging_s,
        save_steps=1000000,
        save_strategy="no",
        save_total_limit=0,
        eval_strategy="no",
        eval_steps=500,
        fp16=True,  # Enable mixed-precision training for faster performance
        report_to=args.report_to,
        remove_unused_columns=True,
        overwrite_output_dir=True,
        weight_decay=0.01,
    )

    # Trainer
    print("Initializing Trainer...")
    loss_logger = BatchLossLogger()
    jinga_callback = ProgressCallback(args.reg_end,args.clip_stage,args.name,args.clip_strategy,args.clip_mode)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[loss_logger,jinga_callback],
        data_collator = datacollator,
    )
    print("Injecting regularization loss in Trainer...")
    jinga_callback.trainer = trainer
    trainer = inject_regularization(trainer,args)

    
    # Train the Model
    print("Starting training...")
    trainer.train()

    if args.report_to == "wandb":
         wandb.log({"loss": loss_logger.losses[-1] if loss_logger.losses else 0})
    
    if args.task == 'MNLI' or args.task == 'COLA' or args.task == 'QNLI' or args.task == 'QQP' or args.task == 'SST' or args.task == 'RTE' or args.task == 'MRPC' or args.task=='STSB' :
        torch.save(model.classifier.state_dict(),f"./outputs/{args.name}/classifier.pth")
        torch.save(model.pooler.state_dict(),f"./outputs/{args.name}/pooler.pth")
        
    elif args.task == 'CIFAR' or args.task == 'FOOD101' or args.task == 'CIFAR100' :
        torch.save(model.classifier.state_dict(),f"./outputs/{args.name}/classifier.pth")

    # Save the Final Model
    print("Saving the LoRA-adapted model...")
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, Linear):
                for adpt in module.active_adapters:
                    module.apply_mask(adpt)
    plot_layer_ranks(model,target_modules,args.name)
    trainer.save_model(f"./outputs/{args.name}")
    print("Model saved successfully!")

    if args.loss=='yes':
        with open(f'./outputs/{args.name}/losses.json','w') as file:
             json.dump(loss_logger.losses,file,indent=4)

if __name__ == "__main__":
    main()
