import wandb

def initialze_wandb(args):
    ''' Initializing wandb and setting the appropriate configuration for NormAL LoRA(jinga) and Original LoRA(no-jinga) '''
    
     wandb.init(
                project = "NormAL LoRA",
                group = args.task,
                name = args.name
            )
    if args.mode == "jinga":
         config = {
            "epochs" : args.epochs,
            "learning_rate" : args.lr,
            "batch_size" : args.batch_s,
            "seed" : args.seed,
            "run" : args.name,
            "model" : args.model,
            "mode" : args.mode,
            "rank" : args.rank,
            "reg_end" : args.reg_end,
            "clip_stage" : args.clip_stage,
            "total_ranks" : args.total_r,
        }
    else:
        config = {
            "epochs" : args.epochs,
            "learning_rate" : args.lr,
            "batch_size" : args.batch_s,
            "seed" : args.seed,
            "run" : args.name,
            "model" : args.model,
            "mode" : args.mode,
            "rank" : args.rank
        }
    wandb.config.update(config)
