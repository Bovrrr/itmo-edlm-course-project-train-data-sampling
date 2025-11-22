import random
import numpy as np
import torch

torch.set_float32_matmul_precision("high")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# 1. LOAD MODEL + TOKENIZER
# ---------------------------

def load_model(model_name: str, num_labels: int = 2):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        trust_remote_code=True
    )
    model.to(DEVICE)
    return model


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# ---------------------------
# 2. BUILD DATALOADER
# ---------------------------

def build_dataloader(dataset, tokenizer, batch_size=32, shuffle=True):

    def collate_fn(batch):
        enc = tokenizer.pad(
            {
                "input_ids": [x["input_ids"] for x in batch],
                "attention_mask": [x["attention_mask"] for x in batch],
            },
            return_tensors="pt",
        )
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        enc["labels"] = labels
        return {k: v.to(DEVICE) for k, v in enc.items()}

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )



# ---------------------------
# 3. ONE TRAIN STEP
# ---------------------------

def train_epoch(model, train_loader, optimizer):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


# ---------------------------
# 4. EVAL STEP
# ---------------------------

@torch.no_grad()
def eval_epoch(model, val_loader, metric_acc, metric_f1):
    model.eval()
    all_preds = []
    all_labels = []

    val_loss = 0.0

    for batch in val_loader:
        outputs = model(**batch)
        val_loss += outputs.loss.item()

        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())

    acc = metric_acc.compute(predictions=all_preds, references=all_labels)["accuracy"]
    f1 = metric_f1.compute(predictions=all_preds, references=all_labels)["f1"]

    return {
        "val_loss": val_loss / len(val_loader),
        "accuracy": acc,
        "f1": f1,
    }


# ---------------------------
# 5. FULL TRAIN LOOP
# ---------------------------

def train_model(
    model_name: str,
    train_dataset,
    val_dataset,
    epochs=3,
    lr=2e-5,
    batch_size=32
):
    import evaluate
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, num_labels=2)

    train_loader = build_dataloader(train_dataset, tokenizer, batch_size=batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, tokenizer, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer)
        metrics = eval_epoch(model, val_loader, metric_acc, metric_f1)

        print(f"\nEpoch {epoch+1}")
        print("train_loss:", train_loss)
        print(metrics)

    return model, metrics
