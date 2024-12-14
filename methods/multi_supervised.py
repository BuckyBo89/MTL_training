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
from sklearn.metrics import precision_score, recall_score, f1_score



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

class MultiTaskModel(torch.nn.Module): 
    def __init__(self, base_model_name, num_labels_task1, num_labels_task2, cache_dir):
        super(MultiTaskModel, self).__init__()
        
        # 初始化基础模型
        self.base_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=num_labels_task1, cache_dir=cache_dir)
        
        # 根据模型类型定义共享的编码器
        if "roberta" in base_model_name.lower():  # 针对 RoBERTa 和 XLM-R
            self.shared_model = self.base_model.roberta
        elif "bert" in base_model_name.lower():  # 针对 mBERT
            self.shared_model = self.base_model.bert
        else:
            raise ValueError("Unsupported model type: {}".format(base_model_name))
        
        self.classifier_task1 = torch.nn.Linear(self.shared_model.config.hidden_size, num_labels_task1)
        self.classifier_task2 = torch.nn.Linear(self.shared_model.config.hidden_size, num_labels_task2)

    def forward(self, input_ids, attention_mask, task='task1'):
        outputs = self.shared_model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] token 的表示
        if task == 'task1':
            logits = self.classifier_task1(sequence_output)
        else:
            logits = self.classifier_task2(sequence_output)
        return logits

@timeit
def run_supervised_experiment(
        data_task1,
        data_task2,
        model_name,
        cache_dir,
        batch_size,
        DEVICE,
        pos_bit=0,
        finetune=False,
        num_labels_task1=0,
        num_labels_task2=0,
        epochs=3,
        save_path=None,
        use_pcgrad=False,  # 添加 use_pcgrad 参数
        **kwargs):
    print(f'Beginning supervised evaluation with {model_name}...')

    # 初始化多任务学习模型
    detector = MultiTaskModel(
        base_model_name=model_name,
        num_labels_task1=num_labels_task1,
        num_labels_task2=num_labels_task2,
        cache_dir=cache_dir
    ).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # 如果提供了预训练模型权重，则加载它
    if ("state_dict_path" in kwargs) and ("state_dict_key" in kwargs):
        detector.load_state_dict(
            torch.load(
                kwargs["state_dict_path"],
                map_location='cpu')[kwargs["state_dict_key"]])

    # 如果需要微调模型
    if finetune:
        fine_tune_multi_task_model(
            model=detector,
            tokenizer=tokenizer,
            data_task1=data_task1,
            data_task2=data_task2,
            batch_size=batch_size,
            DEVICE=DEVICE,
            num_labels_task1=num_labels_task1,
            num_labels_task2=num_labels_task2,
            epochs=epochs,
            save_path=save_path,
            use_pcgrad=use_pcgrad  # 将 use_pcgrad 传递给 fine_tune_multi_task_model
        )

    # 准备数据
    train_text_task1 = data_task1['train']['text']
    train_label_task1 = data_task1['train']['label']
    test_text_task1 = data_task1['test']['text']
    test_label_task1 = data_task1['test']['label']

    train_text_task2 = data_task2['train']['text']
    train_label_task2 = data_task2['train']['label']
    test_text_task2 = data_task2['test']['text']
    test_label_task2 = data_task2['test']['label']

    # 获取任务1和任务2的预测
    train_preds_task1 = get_supervised_model_prediction(detector, tokenizer, train_text_task1, batch_size, DEVICE, pos_bit, task='task1')
    test_preds_task1 = get_supervised_model_prediction(detector, tokenizer, test_text_task1, batch_size, DEVICE, pos_bit, task='task1')

    train_preds_task2 = get_supervised_model_prediction(detector, tokenizer, train_text_task2, batch_size, DEVICE, pos_bit, task='task2')
    test_preds_task2 = get_supervised_model_prediction(detector, tokenizer, test_text_task2, batch_size, DEVICE, pos_bit, task='task2')

    # 计算任务1的指标
    y_train_pred_prob_task1 = train_preds_task1
    y_train_pred_task1 = [round(_) for _ in y_train_pred_prob_task1]
    y_train_task1 = train_label_task1

    y_test_pred_prob_task1 = test_preds_task1
    y_test_pred_task1 = [round(_) for _ in y_test_pred_prob_task1]
    y_test_task1 = test_label_task1

    train_res_task1 = cal_metrics(y_train_task1, y_train_pred_task1, y_train_pred_prob_task1)
    test_res_task1 = cal_metrics(y_test_task1, y_test_pred_task1, y_test_pred_prob_task1)

    # 计算任务2的指标
    y_train_pred_prob_task2 = train_preds_task2
    y_train_pred_task2 = [round(_) for _ in y_train_pred_prob_task2]
    y_train_task2 = train_label_task2

    y_test_pred_prob_task2 = test_preds_task2
    y_test_pred_task2 = [round(_) for _ in y_test_pred_prob_task2]
    y_test_task2 = test_label_task2

    train_res_task2 = cal_metrics(y_train_task2, y_train_pred_task2, y_train_pred_prob_task2)
    test_res_task2 = cal_metrics(y_test_task2, y_test_pred_task2, y_test_pred_prob_task2)

    # 打印任务1和任务2的结果
    acc_train_task1, precision_train_task1, recall_train_task1, f1_train_task1, auc_train_task1 = train_res_task1
    acc_test_task1, precision_test_task1, recall_test_task1, f1_test_task1, auc_test_task1 = test_res_task1
    print(f"Task 1 - {model_name} acc_train: {acc_train_task1}, precision_train: {precision_train_task1}, recall_train: {recall_train_task1}, f1_train: {f1_train_task1}, auc_train: {auc_train_task1}")
    print(f"Task 1 - {model_name} acc_test: {acc_test_task1}, precision_test: {precision_test_task1}, recall_test: {recall_test_task1}, f1_test: {f1_test_task1}, auc_test: {auc_test_task1}")

    acc_train_task2, precision_train_task2, recall_train_task2, f1_train_task2, auc_train_task2 = train_res_task2
    acc_test_task2, precision_test_task2, recall_test_task2, f1_test_task2, auc_test_task2 = test_res_task2
    print(f"Task 2 - {model_name} acc_train: {acc_train_task2}, precision_train: {precision_train_task2}, recall_train: {recall_train_task2}, f1_train: {f1_train_task2}, auc_train: {auc_train_task2}")
    print(f"Task 2 - {model_name} acc_test: {acc_test_task2}, precision_test: {precision_test_task2}, recall_test: {recall_test_task2}, f1_test: {f1_test_task2}, auc_test: {auc_test_task2}")

    # 保存预测结果
    predictions = {
        'task1': {
            'train': train_preds_task1,
            'test': test_preds_task1,
        },
        'task2': {
            'train': train_preds_task2,
            'test': test_preds_task2,
        }
    }

    # 释放GPU内存
    del detector
    torch.cuda.empty_cache()

    return {
        'name': model_name,
        'predictions': predictions,
        'task1': {
            'acc_train': acc_train_task1,
            'precision_train': precision_train_task1,
            'recall_train': recall_train_task1,
            'f1_train': f1_train_task1,
            'auc_train': auc_train_task1,
            'acc_test': acc_test_task1,
            'precision_test': precision_test_task1,
            'recall_test': recall_test_task1,
            'f1_test': f1_test_task1,
            'auc_test': auc_test_task1,
        },
        'task2': {
            'acc_train': acc_train_task2,
            'precision_train': precision_train_task2,
            'recall_train': recall_train_task2,
            'f1_train': f1_train_task2,
            'auc_train': auc_train_task2,
            'acc_test': acc_test_task2,
            'precision_test': precision_test_task2,
            'recall_test': recall_test_task2,
            'f1_test': f1_test_task2,
            'auc_test': auc_test_task2,
        }
    }





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


def prepare_datasets(tokenizer, data_task1, data_task2, batch_size):

    encodings_task1 = tokenizer(data_task1['train']['text'], truncation=True, padding=True)
    train_dataset_task1 = CustomDataset(encodings_task1, data_task1['train']['label'])

    encodings_task1_test = tokenizer(data_task1['test']['text'], truncation=True, padding=True)
    test_dataset_task1 = CustomDataset(encodings_task1_test, data_task1['test']['label'])

    encodings_task2 = tokenizer(data_task2['train']['text'], truncation=True, padding=True)
    train_dataset_task2 = CustomDataset(encodings_task2, data_task2['train']['label'])

    encodings_task2_test = tokenizer(data_task2['test']['text'], truncation=True, padding=True)
    test_dataset_task2 = CustomDataset(encodings_task2_test, data_task2['test']['label'])

    train_loader_task1 = DataLoader(train_dataset_task1, batch_size=batch_size, shuffle=True)
    train_loader_task2 = DataLoader(train_dataset_task2, batch_size=batch_size, shuffle=True)

    test_loader_task1 = DataLoader(test_dataset_task1, batch_size=batch_size)
    test_loader_task2 = DataLoader(test_dataset_task2, batch_size=batch_size)

    return train_loader_task1, train_loader_task2, test_loader_task1, test_loader_task2


def fine_tune_multi_task_model(
        model, tokenizer, data_task1, data_task2, batch_size, DEVICE, num_labels_task1, num_labels_task2, 
        epochs=3, save_path=None, use_pcgrad=False):
    # 准备数据
    train_loader_task1, train_loader_task2, test_loader_task1, test_loader_task2 = prepare_datasets(tokenizer, data_task1, data_task2, batch_size)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    for epoch in range(epochs):
        model.train()
        total_loss_task1, total_loss_task2 = 0, 0

        # 交替训练两个任务
        for (batch_task1, batch_task2) in zip(train_loader_task1, train_loader_task2):
            optimizer.zero_grad()

            # 任务1的前向传播和损失计算
            input_ids_task1 = batch_task1['input_ids'].to(DEVICE)
            attention_mask_task1 = batch_task1['attention_mask'].to(DEVICE)
            labels_task1 = batch_task1['labels'].to(DEVICE)
            outputs_task1 = model(input_ids_task1, attention_mask_task1, task='task1')
            loss_task1 = torch.nn.CrossEntropyLoss()(outputs_task1, labels_task1)
            total_loss_task1 += loss_task1.item()
            loss_task1.backward(retain_graph=True)

            # 任务2的前向传播和损失计算
            input_ids_task2 = batch_task2['input_ids'].to(DEVICE)
            attention_mask_task2 = batch_task2['attention_mask'].to(DEVICE)
            labels_task2 = batch_task2['labels'].to(DEVICE)
            outputs_task2 = model(input_ids_task2, attention_mask_task2, task='task2')
            loss_task2 = torch.nn.CrossEntropyLoss()(outputs_task2, labels_task2)
            total_loss_task2 += loss_task2.item()
            loss_task2.backward()

            # PCGrad修正：对梯度进行投影
            if use_pcgrad and epoch > 0:  # 仅在设置了PCGrad时使用
                with torch.no_grad():
                    # 获取任务1和任务2的梯度并展平
                    grad_task1 = [param.grad.clone().flatten() if param.grad is not None else None for param in model.parameters()]
                    grad_task2 = [param.grad.clone().flatten() if param.grad is not None else None for param in model.parameters()]

                    # 计算投影并更新梯度
                    for g1, g2 in zip(grad_task1, grad_task2):
                        if g1 is not None and g2 is not None:
                            proj = (g1 @ g2) / (g2.norm() ** 2 + 1e-8)  # 计算投影
                            g1.sub_(proj * g2)  # 更新任务1的梯度

                    # 重新设置梯度到模型参数中
                    for param, g in zip(model.parameters(), grad_task1):
                        if g is not None:
                            param.grad = g.view(param.grad.shape)

            # 更新模型参数
            optimizer.step()

        avg_loss_task1 = total_loss_task1 / len(train_loader_task1)
        avg_loss_task2 = total_loss_task2 / len(train_loader_task2)

        print(f'Epoch {epoch + 1}/{epochs} - Loss Task1: {avg_loss_task1}, Loss Task2: {avg_loss_task2}')

        evaluate_multi_task_model(model, test_loader_task1, test_loader_task2, DEVICE)
        # 保存模型
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), f'{save_path}/multi_task_model_epoch{epoch+1}.pt')




def evaluate_multi_task_model(model, test_loader_task1, test_loader_task2, DEVICE):
    model.eval()
    all_labels_task1, all_preds_task1 = [], []
    all_labels_task2, all_preds_task2 = [], []

    # 任务1的评估
    with torch.no_grad():
        for batch_task1 in test_loader_task1:
            input_ids_task1 = batch_task1['input_ids'].to(DEVICE)
            attention_mask_task1 = batch_task1['attention_mask'].to(DEVICE)
            labels_task1 = batch_task1['labels'].to(DEVICE)
            outputs_task1 = model(input_ids_task1, attention_mask_task1, task='task1')
            _, predicted_task1 = torch.max(outputs_task1, 1)
            all_labels_task1.extend(labels_task1.cpu().numpy())
            all_preds_task1.extend(predicted_task1.cpu().numpy())

    # 任务2的评估
    with torch.no_grad():
        for batch_task2 in test_loader_task2:
            input_ids_task2 = batch_task2['input_ids'].to(DEVICE)
            attention_mask_task2 = batch_task2['attention_mask'].to(DEVICE)
            labels_task2 = batch_task2['labels'].to(DEVICE)
            outputs_task2 = model(input_ids_task2, attention_mask_task2, task='task2')
            _, predicted_task2 = torch.max(outputs_task2, 1)
            all_labels_task2.extend(labels_task2.cpu().numpy())
            all_preds_task2.extend(predicted_task2.cpu().numpy())

    # 计算任务1的指标
    precision_task1 = precision_score(all_labels_task1, all_preds_task1, average='weighted')
    recall_task1 = recall_score(all_labels_task1, all_preds_task1, average='weighted')
    f1_task1 = f1_score(all_labels_task1, all_preds_task1, average='weighted')

    # 计算任务2的指标
    precision_task2 = precision_score(all_labels_task2, all_preds_task2, average='weighted')
    recall_task2 = recall_score(all_labels_task2, all_preds_task2, average='weighted')
    f1_task2 = f1_score(all_labels_task2, all_preds_task2, average='weighted')

    print(f'Task1 - Precision: {precision_task1}, Recall: {recall_task1}, F1: {f1_task1}')
    print(f'Task2 - Precision: {precision_task2}, Recall: {recall_task2}, F1: {f1_task2}')