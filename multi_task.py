import argparse
import datetime
import os
import json
import random
import numpy as np
import torch
import dataset_loader_attribution
from methods.utils import load_base_model, load_base_model_and_tokenizer, filter_test_data
from methods.multi_supervised import run_supervised_experiment
from methods.detectgpt import run_perturbation_experiments
from methods.gptzero import run_gptzero_experiment
from methods.metric_based import get_ll, get_rank, get_entropy, get_rank_GLTR, run_threshold_experiment, \
    run_GLTR_experiment

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


if __name__ == '__main__':
    # 在主函数开始时立即设置随机种子
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset1', type=str, default="Essay")
    parser.add_argument('--dataset2', type=str, default="Essay")
    parser.add_argument('--method', type=str, default="Log-Likelihood")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num_labels_task1', type=int, required=True)
    parser.add_argument('--num_labels_task2', type=int, required=True)
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-base")
    parser.add_argument('--cache_dir', type=str, default=".cache")
    parser.add_argument('--DEVICE', type=str, default="cuda")
    parser.add_argument('--use_pcgrad', action='store_true', help="Enable PCGrad during training")

    # params for DetectGPT
    parser.add_argument('--pct_words_masked', type=float, default=0.3)
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_perturbation_list', type=str, default="10")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--random_fills', action='store_true')
    parser.add_argument('--random_fills_tokens', action='store_true')

    # params for GPTZero
    parser.add_argument('--gptzero_key', type=str, default="")

    args = parser.parse_args()

    DEVICE = args.DEVICE

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

    print(f'Loading dataset {args.dataset1}...')
    file_path1 = os.path.join('file_to_multitude', f'{args.dataset1}.csv')
    print(f'Loading dataset {args.dataset2}...')
    file_path2 = os.path.join('file_to_multitude', f'{args.dataset2}.csv')

    data_task1 = dataset_loader_attribution.load_custom_dataset_task1(file_path1)
    data_task2 = dataset_loader_attribution.load_custom_dataset_task2(file_path2)

    base_model_name = args.base_model_name.replace('/', '_')

    SAVE_PATH = f'file_to_update_results/{args.dataset1}-{args.dataset2}'
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_PATH)}")

    # write args to file
    with open(os.path.join(SAVE_PATH, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    mask_filling_model_name = args.mask_filling_model_name
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    n_perturbation_rounds = args.n_perturbation_rounds
    n_similarity_samples = args.n_similarity_samples

    cache_dir = args.cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    # get generative model
    base_model, base_tokenizer = load_base_model_and_tokenizer(
        args.base_model_name, cache_dir)
    load_base_model(base_model, DEVICE)


    def ll_criterion(text):
        return get_ll(
            text, base_model, base_tokenizer, DEVICE)
    def rank_criterion(text):
        return -get_rank(text,
                         base_model, base_tokenizer, DEVICE, log=False)
    def logrank_criterion(text):
        return -get_rank(text,
                         base_model, base_tokenizer, DEVICE, log=True)
    def entropy_criterion(text):
        return get_entropy(
            text, base_model, base_tokenizer, DEVICE)
    def GLTR_criterion(text):
        return get_rank_GLTR(
            text, base_model, base_tokenizer, DEVICE)
    outputs = []

    if args.method == "mBERT":
        outputs.append(
            run_supervised_experiment(
                data_task1,
                data_task2,
                model_name='bert-base-multilingual-cased',
                cache_dir=cache_dir,
                batch_size=batch_size,
                DEVICE=DEVICE,
                pos_bit=1,
                finetune=True,
                num_labels_task1=args.num_labels_task1,
                num_labels_task2=args.num_labels_task2,
                epochs=args.epochs,
                save_path=SAVE_PATH + f"/mBERT-{args.epochs}",
                use_pcgrad=args.use_pcgrad
            )
        )
    elif args.method == "XLM-R":
        outputs.append(
            run_supervised_experiment(
                data_task1,
                data_task2,
                model_name='xlm-roberta-base',
                cache_dir=cache_dir,
                batch_size=batch_size,
                DEVICE=DEVICE,
                pos_bit=1,
                finetune=True,
                num_labels_task1=args.num_labels_task1,
                num_labels_task2=args.num_labels_task2,
                epochs=args.epochs,
                save_path=SAVE_PATH + f"/XLM-R-{args.epochs}",
                use_pcgrad=args.use_pcgrad
            )
        )

    # save results
    import pickle as pkl

    with open(os.path.join(SAVE_PATH, f"{args.method}_{args.epochs}_attribution_benchmark_results.pkl"), "wb") as f:
        pkl.dump(outputs, f)

    if not os.path.exists("logs/"):
        os.makedirs("logs/")

    with open("logs/performance_attribution.csv", "a") as wf:
        for row in outputs:
            wf.write(
                f"{args.dataset1},{args.base_model_name},{args.method},{args.epochs},{json.dumps(row['general'])}\n")

    print("Finish")
