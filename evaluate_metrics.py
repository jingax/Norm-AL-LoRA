import os
import subprocess
import argparse

folder_path = "eval"

parser = argparse.ArgumentParser()
parser.add_argument("--exp",type="str",help="exp to run")
parser.add_argument("--name",type="str",help="folder name")
parser.add_argument("--split",type="str",help="split name")
parser.add_argument("--model",type="str",default="None",help="model type")
args = parser.parse_args()

def run_scripts(script_path):
    subprocess.run(["python",script_path,'--exp',args.exp,'--split',args.split,'--name',args.name,'--model',args.model])
def main():
    script_path=os.path.join(folder_path,f"eval_{args.exp}.py")
    run_scripts(script_path)

if __name__ == "__main__":
    main()
