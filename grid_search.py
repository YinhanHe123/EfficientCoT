import argparse
import json
import logging
import random
import subprocess
import sys

def get_max_acc(file):
    with open(file) as f:
        lines = f.readlines()
    if len(lines) == 0:
        return 0  
    return max([json.loads(l)['numerical_accuracy'] for l in lines[-5:]])

parser = argparse.ArgumentParser(description="Contemplation Tokens with Reasoning Ability")
parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "svamp", "multiarith", "commonsense_qa", "coin_flip"],
                    help="Dataset to use")
parser.add_argument("--config", type=str, default="small", choices=["small", "mistral"],
                    help="Model config to use")
parser.add_argument("--device", type=str, default="0")
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(f"{args.dataset}_{args.config}.log"), logging.StreamHandler(sys.stdout)],
)

logging.info("Run grid search")

linears = [0.01, 0.001, 0.0001]
llmlrs = [1e-5, 1e-7]
llmwds = [1e-3, 1e-5]
linear_epochs = [1,3,5]
llm_epochs = [1,2]

res_path = f"/home/nee7ne/EfficientCoT/results/effi_cot/vanilla/{args.config}/{args.dataset}/evaluation_results.jsonl"
max_acc = 0
combinations = []
for llr in linears:
    for lwd in linears:
        for llmlr in llmlrs:
            for llmwd in llmwds:
                for le in linear_epochs:
                    for llme in llm_epochs:
                        combinations.append((llr, lwd, llmlr, llmwd, le, llme))

for (st_llr, st_lwd, st_llmlr, st_llmwd, st_le, st_llme) in random.sample(combinations, len(combinations)):
    train_st = True
    early_stop, max_tries = 0, 0
    logging.info(f"Training sentence transformer - llr = {st_llr} | lwd = {st_lwd} | le = {st_le} | llmlr = {st_llmlr} | llmwd = {st_llmwd} | llme = {st_llme}")
    for (cg_llr, cg_lwd, cg_llmlr, cg_llmwd, cg_le, cg_llme) in random.sample(combinations, len(combinations)):
        logging.info(f"Training contemp generator - llr = {cg_llr} | lwd = {cg_lwd} | le = {cg_le} | llmlr = {cg_llmlr} | llmwd = {cg_llmwd} | llme = {cg_llme}")
        return_code = 1
        while return_code != 0:
            result = subprocess.run(['python', 'main.py', '--config', args.config, '--mode', 'effi_cot', '--dataset', args.dataset, '--device', args.device, '--variation', 'vanilla', '-stllr', str(st_llr), '-stlwd', str(st_lwd), '-stle', str(st_le), '-stllmlr', str(st_llmlr), '-stllmwd', str(st_llmwd), '-stllme', str(st_llme), '-cgllr', str(cg_llr), '-cglwd', str(cg_lwd), '-cgle', str(cg_le), '-cgllmlr', str(cg_llmlr), '-cgllmwd', str(cg_llmwd), '-cgllme', str(cg_llme), '--num_exps', '1'], capture_output=True, text=True)
            return_code = result.returncode
        if train_st:
            train_st = False
        acc = get_max_acc(res_path)
        if acc > max_acc:
            max_acc = acc
            logging.info(f"Found best acc = {acc} | stllr = {st_llr} | stlwd = {st_lwd} | stle = {st_le} | stllmlr = {st_llmlr} | stllmwd = {st_llmwd} | stllme = {st_llme} | cgllr = {cg_llr} | cgle = {cg_le} | cglwd = {cg_lwd} | cgllmlr = {cg_llmlr} | cgllmwd = {cg_llmwd} | cgllme = {cg_llme}")
            logging.info("Python command - " + " ".join(['python', 'main.py', '--config', args.config, '--mode', 'effi_cot', '--dataset', args.dataset, '--device', args.device, '--variation', 'vanilla', '-stllr', str(st_llr), '-stlwd', str(st_lwd), '-stle', str(st_le), '-stllmlr', str(st_llmlr), '-stllmwd', str(st_llmwd), '-stllme', str(st_llme), '-cgllr', str(cg_llr), '-cglwd', str(cg_lwd), '-cgle', str(cg_le), '-cgllmlr', str(cg_llmlr), '-cgllmwd', str(cg_llmwd), '-cgllme', str(cg_llme), '--num_exps', '1']))
        elif (max_acc - acc) > 0.05:
            early_stop += 1
        max_tries += 1
        if early_stop > 5:
            break
        elif max_tries > 10:
            break

# for st_llr in linear:
#     for st_lwd in linear:
#         for st_llmlr in llmlr:
#             for st_llmwd in llmwd:
#                 for st_le in linear_epochs:
#                     for st_llme in llm_epochs:
#                         train_st = True
#                         logging.info(f"Training sentence transformer - llr = {st_llr} | lwd = {st_lwd} | llmlr = {st_llmlr} | llmwd = {st_llmwd}")
#                         for cg_llr in linear:
#                             for cg_lwd in linear:
#                                 for cg_llmlr in llmlr:
#                                     for cg_llmwd in llmwd:
#                                         for cg_le in linear_epochs:
#                                             for cg_llme in llm_epochs:
#                                                 logging.info(f"Training contemp generator - llr = {cg_llr} | lwd = {cg_lwd} | llmlr = {cg_llmlr} | llmwd = {cg_llmwd}")
#                                                 return_code = 1
#                                                 while return_code != 0:
#                                                     result = subprocess.run(['python', 'main.py', '--config', args.config, '--mode', 'effi_cot', '--dataset', args.dataset, '--device', args.device, '--variation', 'vanilla', '-stllr', str(st_llr), '-stlwd', str(st_lwd), '-stle', str(st_le), '-stllmlr', str(st_llmlr), '-stllmwd', str(st_llmwd), '-stllme', str(st_llme), '-cgllr', str(cg_llr), '-cglwd', str(cg_lwd), '-cgle', str(cg_le), '-cgllmlr', str(cg_llmlr), '-cgllmwd', str(cg_llmwd), '-cgllme', str(cg_llme), '-train_st', str(train_st), '--num_exps', '1'], capture_output=True, text=True)
#                                                     return_code = result.returncode
#                                                 if train_st:
#                                                     train_st = False
#                                                 acc = get_max_acc(res_path)
#                                                 if acc > max_acc:
#                                                     max_acc = acc
#                                                 elif 
# for st_llr in linear:
#     for st_lwd in linear:
#         for st_llmlr in llmlr:
#             for st_llmwd in llmwd:
#                 train_st = True
#                 logging.info(f"Training sentence transformer - llr = {st_llr} | lwd = {st_lwd} | llmlr = {st_llmlr} | llmwd = {st_llmwd}")
#                 return_code = 1
#                 while return_code != 0:
#                     result = subprocess.run(['python', 'main.py', '--config',args.config, '--mode', 'effi_cot', '--dataset', args.dataset, '--device', args.device, '--variation', 'vanilla', '-stllr', str(st_llr), '-stlwd', str(st_lwd), '-stle', '5', '-stllmlr', str(st_llmlr), '-stllmwd', str(st_llmwd), '-stllme', '2', '-cgllr', "0.01", '-cglwd', "0.01", '-cgle', '5', '-cgllmlr', "1e-07", '-cgllmwd', "0.01", '-cgllme', '2', '-train_st', str(train_st), '--num_exps', '1'], capture_output=True, text=True)
#                     return_code = result.returncode
# for st_llme in epochs:
#     for st_le in epochs:
#         train_st = True
#         logging.info(f"Training sentence transformer - st_le = {st_le} | st_llme = {st_llme}")
#         for cg_le in epochs:
#             for cg_llme in epochs:
#                 logging.info(f"Training contemp generator - cg_le = {cg_le} | cg_llme = {cg_llme}")
#                 return_code = 1
#                 while return_code != 0:
#                     result = subprocess.run(['python', 'main.py', '--config',args.config, '--mode', 'effi_cot', '--dataset', args.dataset, '--device', args.device, '--variation', 'vanilla', '-stle', str(st_le), '-stllme', str(st_llme), '-cgle', str(cg_le), '-cgllme', str(cg_llme), '-train_st', str(train_st), '--num_exps', '1'], capture_output=True, text=True)
#                     return_code = result.returncode
#                 if train_st:
#                     train_st = False