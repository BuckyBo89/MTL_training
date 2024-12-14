import random
import datasets
import tqdm
import pandas as pd
import re

# you can add more datasets here and write your own dataset parsing function


def process_spaces(text):
    return text.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def process_text_truthfulqa_adv(text):

    if "I am sorry" in text:
        first_period = text.index('.')
        start_idx = first_period + 2
        text = text[start_idx:]
    if "as an AI language model" in text or "As an AI language model" in text:
        first_period = text.index('.')
        start_idx = first_period + 2
        text = text[start_idx:]
    return text


def load_TruthfulQA(cache_dir):
    f = pd.read_csv("datasets/TruthfulQA_LLMs.csv")
    q = f['Question'].tolist()
    a_human = f['Best Answer'].tolist()
    mgt_text_list = []
    for detectLLM in [
        "ChatGPT",
        "ChatGLM",
        "Dolly",
        "ChatGPT-turbo",
        "GPT4",
            "StableLM"]:
        mgt_text_list.append(f[f'{detectLLM}_answer'].fillna("").tolist())
    c = f['Category'].tolist()

    res = []
    for i in range(len(q)):
        if len(a_human[i].split()) <= 1:
            continue
        flag = 1
        for mgt_text in mgt_text_list:
            if len(mgt_text[i].split()) <= 1 or len(mgt_text[i]) >= 2000:
                flag = 0
                break
        if flag:
            res.append([q[i],
                        a_human[i],
                        mgt_text_list[0][i],
                        mgt_text_list[1][i],
                        mgt_text_list[2][i],
                        mgt_text_list[3][i],
                        mgt_text_list[4][i],
                        mgt_text_list[5][i],
                        c[i]])

    data_new = {
        'train': {
            'text': [],
            'label': [],
            'category': [],
        },
        'test': {
            'text': [],
            'label': [],
            'category': [],
        }

    }

    index_list = list(range(len(res)))
    random.seed(0)
    random.shuffle(index_list)

    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * 0.8:
            data_partition = 'train'
        else:
            data_partition = 'test'

        for j in range(1, 8):
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][j]))
            data_new[data_partition]['label'].append(j - 1)

    return data_new


def load_SQuAD1(cache_dir):
    f = pd.read_csv("datasets/SQuAD1_LLMs.csv")
    q = f['Question'].tolist()
    a_human = [eval(_)['text'][0] for _ in f['answers'].tolist()]
    mgt_text_list = []
    for detectLLM in [
        "ChatGPT",
        "ChatGLM",
        "Dolly",
        "ChatGPT-turbo",
        "GPT4",
            "StableLM"]:
        mgt_text_list.append(f[f'{detectLLM}_answer'].fillna("").tolist())

    res = []
    for i in range(len(q)):
        if len(a_human[i].split()) <= 1:
            continue
        flag = 1
        for mgt_text in mgt_text_list:
            if len(mgt_text[i].split()) <= 1:
                flag = 0
                break
        if flag:
            res.append([q[i],
                        a_human[i],
                        mgt_text_list[0][i],
                        mgt_text_list[1][i],
                        mgt_text_list[2][i],
                        mgt_text_list[3][i],
                        mgt_text_list[4][i],
                        mgt_text_list[5][i]])

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }

    }

    index_list = list(range(len(res)))
    random.seed(0)
    random.shuffle(index_list)

    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * 0.8:
            data_partition = 'train'
        else:
            data_partition = 'test'

        for j in range(1, 8):
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][j]))
            data_new[data_partition]['label'].append(j - 1)
    return data_new


def load_NarrativeQA(cache_dir):
    f = pd.read_csv("datasets/NarrativeQA_LLMs.csv")
    q = f['Question'].tolist()
    a_human = f['answers'].tolist()
    a_human = [_.split(";")[0] for _ in a_human]
    mgt_text_list = []
    for detectLLM in [
        "ChatGPT",
        "ChatGLM",
        "Dolly",
        "ChatGPT-turbo",
        "GPT4",
            "StableLM"]:
        mgt_text_list.append(f[f'{detectLLM}_answer'].fillna("").tolist())

    res = []
    for i in range(len(q)):
        if len(a_human[i].split()) <= 1 or len(a_human[i].split()) >= 150:
            continue
        flag = 1
        for mgt_text in mgt_text_list:
            if len(
                    mgt_text[i].split()) <= 1 or len(
                    mgt_text[i].split()) >= 150:
                flag = 0
                break
        if flag:
            res.append([q[i],
                        a_human[i],
                        mgt_text_list[0][i],
                        mgt_text_list[1][i],
                        mgt_text_list[2][i],
                        mgt_text_list[3][i],
                        mgt_text_list[4][i],
                        mgt_text_list[5][i]])

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }

    }

    index_list = list(range(len(res)))
    random.seed(0)
    random.shuffle(index_list)

    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * 0.8:
            data_partition = 'train'
        else:
            data_partition = 'test'
        for j in range(1, 8):
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][j]))
            data_new[data_partition]['label'].append(j - 1)
    return data_new


def load(name):
    f = pd.read_csv(f"/content/drive/MyDrive/MGTBench/datasets/{name}_LLMs.csv")
    a_human = f["human"].tolist()

    mgt_text_list = []
    # we do not consider chatgpt in this case
    for detectLLM in [
        "ChatGLM",
        "Dolly",
        "ChatGPT-turbo",
        "GPT4All",
        "StableLM",
            "Claude"]:
        mgt_text_list.append(f[f'{detectLLM}'].fillna("").tolist())

    res = []
    for i in range(len(a_human)):
        flag = 1
        if len(a_human[i].split()) <= 1:
            flag = 0
        for mgt_text in mgt_text_list:
            if len(mgt_text[i].split()) <= 1:
                flag = 0
                break
        if flag:
            res.append([a_human[i],
                        mgt_text_list[0][i],
                        mgt_text_list[1][i],
                        mgt_text_list[2][i],
                        mgt_text_list[3][i],
                        mgt_text_list[4][i],
                        mgt_text_list[5][i]])

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }

    }

    index_list = list(range(len(res)))
    random.seed(0)
    random.shuffle(index_list)

    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * 0.8:
            data_partition = 'train'
        else:
            data_partition = 'test'
        for j in range(0, 7):
            data_new[data_partition]['text'].append(
                (res[index_list[i]][j]))
            data_new[data_partition]['label'].append(j)
    return data_new


def load_custom_dataset_task2(file_path):
    f = pd.read_csv(file_path)
    f = f.drop(columns=["language", "length", "source", "label"])
    f['multi_label_encoded'], unique_labels = pd.factorize(f['multi_label'])
    # 打乱数据索引
    f = f.sample(frac=1, random_state=0).reset_index(drop=True)
    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }
    }
    for i in tqdm.tqdm(range(len(f)), desc="Parsing data"):
        data_partition = f.iloc[i]['split']
        if data_partition not in ['train', 'test']:
            continue
        data_new[data_partition]['text'].append((f.iloc[i]['text']))
        data_new[data_partition]['label'].append(f.iloc[i]['multi_label_encoded'])

    return data_new


def load_custom_dataset_task1(file_path):
    f = pd.read_csv(file_path)
    f = f.drop(columns=["language", "length", "source", "label"])

    f['multi_label_encoded'] = f['multi_label'].apply(lambda x: 1 if x == 'human' else 0)

    f = f.sample(frac=1, random_state=0).reset_index(drop=True)

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }
    }

    for i in tqdm.tqdm(range(len(f)), desc="Parsing data"):
        data_partition = f.iloc[i]['split']
        if data_partition not in ['train', 'test']:
            continue
        data_new[data_partition]['text'].append(f.iloc[i]['text'])
        data_new[data_partition]['label'].append(f.iloc[i]['multi_label_encoded'])

    return data_new



def load_identify_model_dataset(file_path):
    f = pd.read_csv(file_path)
    f = f.drop(columns=["language", "length", "source", "label"])

    # 根据条件对multi_label进行编码
    f['multi_label_encoded'] = f['multi_label'].apply(lambda x: 1 if x == 'vicuna-13b' else 0)

    # 打乱数据索引
    f = f.sample(frac=1, random_state=0).reset_index(drop=True)

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }
    }

    for i in tqdm.tqdm(range(len(f)), desc="Parsing data"):
        data_partition = f.iloc[i]['split']
        if data_partition not in ['train', 'test']:
            continue
        data_new[data_partition]['text'].append(f.iloc[i]['text'])
        data_new[data_partition]['label'].append(f.iloc[i]['multi_label_encoded'])

    return data_new
