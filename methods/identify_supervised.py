import numpy as np
import transformers
import torch
from tqdm import tqdm
from methods.utils import timeit, cal_metrics
from torch.utils.data import DataLoader
from transformers import AdamW, GPT2ForSequenceClassification, GPT2Tokenizer, AutoModelForSequenceClassification, \
    AutoTokenizer
import matplotlib.pyplot as plt
import os


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


@timeit
def run_supervised_experiment(
        data,
        model,
        cache_dir,
        batch_size,
        DEVICE,
        pos_bit=0,
        finetune=False,
        num_labels=2,
        epochs=3,
        save_path=None,
        **kwargs):
    print(f'Beginning supervised evaluation with {model}...')

    # 根据模型名称选择合适的模型类
    if model == 'gpt2':
        detector = GPT2ForSequenceClassification.from_pretrained(
            model,
            num_labels=num_labels,
            cache_dir=cache_dir).to(DEVICE)

        tokenizer = GPT2Tokenizer.from_pretrained(
            model, cache_dir=cache_dir)
        # 添加填充标记
        # tokenizer.pad_token = tokenizer.eos_token
    else:
        detector = AutoModelForSequenceClassification.from_pretrained(
            model,
            num_labels=num_labels,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=True).to(DEVICE)

        tokenizer = AutoTokenizer.from_pretrained(
            model, cache_dir=cache_dir)

    if ("state_dict_path" in kwargs) and ("state_dict_key" in kwargs):
        detector.load_state_dict(
            torch.load(
                kwargs["state_dict_path"],
                map_location='cpu')[
                kwargs["state_dict_key"]])

    if finetune:
        fine_tune_model(detector, tokenizer, data, batch_size, DEVICE, pos_bit, num_labels, epochs=epochs,
                        save_path=save_path)
        if save_path:
            detector.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    train_text = data['train']['text']
    train_label = data['train']['label']
    test_text = data['test']['text']
    test_label = data['test']['label']

    if num_labels == 2:
        train_preds = get_supervised_model_prediction(detector, tokenizer, train_text, batch_size, DEVICE, pos_bit)
        test_preds = get_supervised_model_prediction(detector, tokenizer, test_text, batch_size, DEVICE, pos_bit)
    else:
        train_preds = get_supervised_model_prediction_multi_classes(detector, tokenizer, train_text, batch_size, DEVICE,
                                                                    pos_bit)
        test_preds = get_supervised_model_prediction_multi_classes(detector, tokenizer, test_text, batch_size, DEVICE,
                                                                   pos_bit)

    predictions = {
        'train': train_preds,
        'test': test_preds,
    }
    y_train_pred_prob = train_preds
    y_train_pred = [round(_) for _ in y_train_pred_prob]
    y_train = train_label

    y_test_pred_prob = test_preds
    y_test_pred = [round(_) for _ in y_test_pred_prob]
    y_test = test_label

    train_res = cal_metrics(y_train, y_train_pred, y_train_pred_prob)
    test_res = cal_metrics(y_test, y_test_pred, y_test_pred_prob)
    acc_train, precision_train, recall_train, f1_train, auc_train = train_res
    acc_test, precision_test, recall_test, f1_test, auc_test = test_res
    print(
        f"{model} acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}")
    print(
        f"{model} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")

    # free GPU memory
    del detector
    torch.cuda.empty_cache()

    return {
        'name': model,
        'predictions': predictions,
        'general': {
            'acc_train': acc_train,
            'precision_train': precision_train,
            'recall_train': recall_train,
            'f1_train': f1_train,
            'auc_train': auc_train,
            'acc_test': acc_test,
            'precision_test': precision_test,
            'recall_test': recall_test,
            'f1_test': f1_test,
            'auc_test': auc_test,
        }
    }


def run_supervised_experiment_multi_test_length(
        data,
        model,
        cache_dir,
        batch_size,
        DEVICE,
        pos_bit=0,
        finetune=False,
        num_labels=2,
        epochs=3,
        save_path=None,
        lengths=[
            10,
            20,
            50,
            100,
            200,
            500,
            -1],
        **kwargs):
    print(f'Beginning supervised evaluation with {model}...')
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(
        model,
        num_labels=num_labels,
        cache_dir=cache_dir,
        ignore_mismatched_sizes=True).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model, cache_dir=cache_dir)

    if ("state_dict_path" in kwargs) and ("state_dict_key" in kwargs):
        detector.load_state_dict(
            torch.load(
                kwargs["state_dict_path"],
                map_location='cpu')[
                kwargs["state_dict_key"]])

    if finetune:
        fine_tune_model(detector, tokenizer, data, batch_size,
                        DEVICE, pos_bit, num_labels, epochs=epochs)
        if save_path:
            detector.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    res = {}
    from methods.utils import cut_length, cal_metrics
    for length in lengths:
        test_text = data['test']['text']
        test_text = [cut_length(_, length) for _ in test_text]
        test_label = data['test']['label']

        if num_labels == 2:
            test_preds = get_supervised_model_prediction(
                detector, tokenizer, test_text, batch_size, DEVICE, pos_bit)
        else:
            test_preds = get_supervised_model_prediction_multi_classes(
                detector, tokenizer, test_text, batch_size, DEVICE, pos_bit)

        y_test_pred_prob = test_preds
        y_test_pred = [round(_) for _ in y_test_pred_prob]
        y_test = test_label

        test_res = cal_metrics(y_test, y_test_pred, y_test_pred_prob)
        acc_test, precision_test, recall_test, f1_test, auc_test = test_res

        print(
            f"{model} {length} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")
        res[length] = test_res

    # free GPU memory
    del detector
    torch.cuda.empty_cache()
    return res


def get_supervised_model_prediction(
        model,
        tokenizer,
        data,
        batch_size,
        DEVICE,
        pos_bit=0):
    with torch.no_grad():
        # get predictions for real
        preds = []
        for start in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]
            batch_data = tokenizer(
                batch_data,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt").to(DEVICE)
            preds.extend(model(**batch_data).logits.softmax(-1)
                         [:, pos_bit].tolist())
    return preds


def get_supervised_model_prediction_multi_classes(
        model, tokenizer, data, batch_size, DEVICE, pos_bit=0):
    with torch.no_grad():
        # get predictions for real
        preds = []
        for start in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]
            batch_data = tokenizer(
                batch_data,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt").to(DEVICE)
            preds.extend(torch.argmax(
                model(**batch_data).logits, dim=1).tolist())
    return preds


import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt


def fine_tune_model(
        model,
        tokenizer,
        data,
        batch_size,
        DEVICE,
        pos_bit=1,
        num_labels=2,
        epochs=3,
        save_path=None,  # 将 save_path 设置为可选参数
        seed=42):
    # 使用生成器设置随机种子，确保每次打乱的顺序相同
    g = torch.Generator()
    g.manual_seed(seed)

    train_text = data['train']['text']
    train_label = data['train']['label']
    test_text = data['test']['text']
    test_label = data['test']['label']

    if pos_bit == 0 and num_labels == 2:
        train_label = [1 if _ == 0 else 0 for _ in train_label]
        test_label = [1 if _ == 0 else 0 for _ in test_label]

    train_encodings = tokenizer(train_text, truncation=True, padding=True)
    test_encodings = tokenizer(test_text, truncation=True, padding=True)
    train_dataset = CustomDataset(train_encodings, train_label)
    test_dataset = CustomDataset(test_encodings, test_label)

    model.train()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    train_acc_list = []
    test_acc_list = []

    best_acc = 0.0  # 用于跟踪最高的测试准确率
    best_model_weights = None  # 用于保存最佳模型权重

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch in tqdm(train_loader, desc=f"Fine-tuning: {epoch} epoch"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Calculate training accuracy
        model.eval()
        with torch.no_grad():
            train_preds_prob = get_supervised_model_prediction_multi_classes(model, tokenizer, train_text, batch_size,
                                                                             DEVICE, pos_bit)
            test_preds_prob = get_supervised_model_prediction_multi_classes(model, tokenizer, test_text, batch_size,
                                                                            DEVICE, pos_bit)

            # Convert probabilities to binary predictions if needed
            train_preds_binary = [round(p) for p in train_preds_prob]
            test_preds_binary = [round(p) for p in test_preds_prob]

            # Calculate metrics
            train_res = cal_metrics(train_label, train_preds_binary, train_preds_prob)
            test_res = cal_metrics(test_label, test_preds_binary, test_preds_prob)

            acc_train, precision_train, recall_train, f1_train, auc_train = train_res
            acc_test, precision_test, recall_test, f1_test, auc_test = test_res

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss}, "
                  f"Train Accuracy: {acc_train}, Precision: {precision_train}, Recall: {recall_train}, F1: {f1_train}, AUC: {auc_train}, "
                  f"Test Accuracy: {acc_test}, Precision: {precision_test}, Recall: {recall_test}, F1: {f1_test}, AUC: {auc_test}")

            train_acc_list.append(acc_train)
            test_acc_list.append(acc_test)

            # 保存最佳模型
            if acc_test > best_acc:
                best_acc = acc_test
                best_model_weights = model.state_dict()

                if save_path:
                    os.makedirs(save_path, exist_ok=True)

                    torch.save(best_model_weights, f"{save_path}/best_model_weights.pt")
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"New best model saved with accuracy {best_acc} at epoch {epoch + 1}")

    # Plotting accuracy curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_acc_list, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), test_acc_list, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_plot.png')
    plt.show()

    # 加载最佳模型权重
    if best_model_weights:
        model.load_state_dict(best_model_weights)
        print("Best model loaded for further evaluation or inference.")