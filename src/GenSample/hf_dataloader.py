from datasets import load_dataset

dataset = load_dataset("imagefolder",
                       data_dir="xxx",
                       drop_labels=True)

dataset.push_to_hub("xxx/xxx")