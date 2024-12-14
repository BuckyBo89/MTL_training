
1.Fine-tune a single task using single_task.py:

python single_task.py --task task1 --dataset dataset_ca_human_train --method "mBERT" --batch_size 16 --epochs 5 --num_labels 2 
python single_task.py --task task2 --dataset dataset_ca_train_new --method "XLM-R" --batch_size 16 --epochs 5 --num_labels 8 


2.Multi-Task Learning Fine-Tuning Example

python multi_task.py --dataset1 dataset_ca_human_train --dataset2 dataset_ca_train_new --method mBERT --batch_size 16 --epochs 5 --num_labels_task1 5 --num_labels_task2 5 

3.Generating model generalization

python model_generalization.py --dataset dataset_en_opt-66b  --method mBERT  --epochs 25
