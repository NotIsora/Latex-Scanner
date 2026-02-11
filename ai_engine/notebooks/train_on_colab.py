# %% [markdown]
# 🚀 Antigravity Math Solver - Training Notebook (Kaggle/Colab Version)
# Dataset: Neeze/CROHME-full (HuggingFace)

# %% [markdown]
# ## 1. Cài đặt thư viện

# %%
!pip install -q transformers datasets evaluate jiwer torch torchvision

# %% [markdown]
# ## 2. Khai báo thư viện & Cấu hình

# %%
import torch
from torch.utils.data import Dataset
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    TrOCRProcessor
)
from datasets import load_dataset
from PIL import Image
import numpy as np

# === CẤU HÌNH ===
# Option 1: 'custom_vit_gpt2' (Encoder: ViT, Decoder: GPT2 from scratch)
# Option 2: 'trocr_finetune' (Tiếp tục train trên model TrOCR của Microsoft) -> KHUYẾN NGHỊ dùng cái này cho nhanh
MODEL_TYPE = "trocr_finetune"

if MODEL_TYPE == "custom_vit_gpt2":
    encoder_checkpoint = "google/vit-base-patch16-224-in21k"
    decoder_checkpoint = "gpt2"
else:
    # Model pre-trained tốt nhất cho viết tay
    encoder_checkpoint = "microsoft/trocr-base-handwritten" 
    decoder_checkpoint = "microsoft/trocr-base-handwritten"

max_length = 128
batch_size = 8
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device} | Mode: {MODEL_TYPE}")

# %% [markdown]
# ## 3. Tải Dữ liệu từ Hugging Face
# Dataset: https://huggingface.co/datasets/Neeze/CROHME-full

# %%
print("⏳ Đang tải dataset Neeze/CROHME-full...")
dataset = load_dataset("Neeze/CROHME-full")

# Kiểm tra cấu trúc dataset
print(f"Dataset structure: {dataset}")
print(f"Sample: {dataset['train'][0]}")

# Nếu dataset chưa chia train/test, ta tự chia
if "test" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.1)

print(f"Train size: {len(dataset['train'])} | Validation size: {len(dataset['test'])}")

# %% [markdown]
# ## 4. Xử lý dữ liệu (Preprocessing)

# %%
# Load Processor (bao gồm Image Processor và Tokenizer)
if MODEL_TYPE == "trocr_finetune":
    processor = TrOCRProcessor.from_pretrained(encoder_checkpoint)
    image_processor = processor.image_processor
    tokenizer = processor.tokenizer
else:
    image_processor = ViTImageProcessor.from_pretrained(encoder_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

# Hàm xử lý từng mẫu dữ liệu
def preprocess_function(examples):
    # 1. Xử lý ảnh: Chuyển sang RGB và qua image_processor
    # Lưu ý: 'image' cột trong dataset là PIL Image
    images = [img.convert("RGB") for img in examples["image"]]
    pixel_values = image_processor(images, return_tensors="pt").pixel_values
    
    # 2. Xử lý text: Tokenize label
    # Lưu ý: Cần kiểm tra tên cột chứa label LaTeX (thường là 'latex' hoặc 'label' hoặc 'text')
    # Ở đây ta thử lấy cột 'latex', nếu không có thì lấy 'text'
    text_column = "latex" if "latex" in examples else "text"
    if text_column not in examples:
        # Fallback tìm cột chứa chuỗi
        available = list(examples.keys())
        text_column = [k for k in available if k != 'image'][0]
    
    texts = examples[text_column]
    
    model_inputs = tokenizer(
        texts, 
        padding="max_length", 
        max_length=max_length, 
        truncation=True
    )
    
    # Gán -100 cho pad token để không tính loss
    labels = model_inputs.input_ids
    labels_with_ignore_index = []
    for label_example in labels:
        label_example = [label if label != tokenizer.pad_token_id else -100 for label in label_example]
        labels_with_ignore_index.append(label_example)
    
    model_inputs["pixel_values"] = pixel_values
    model_inputs["labels"] = labels_with_ignore_index
    
    return model_inputs

# Áp dụng map function để xử lý toàn bộ dataset
# batched=True giúp xử lý nhanh hơn
print("⏳ Đang tiền xử lý dữ liệu...")
processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
processed_dataset.set_format(type="torch")

print("✅ Đã xử lý xong!")

# %% [markdown]
# ## 5. Khởi tạo & Train Model

# %%
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_checkpoint, 
    decoder_checkpoint
)

# Cấu hình token đặc biệt
model.config.decoder_start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# Cấu hình sinh text (Beam Search)
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = max_length
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

# Định nghĩa metric CER (Character Error Rate)
import evaluate
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    # Giải mã
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

# Cấu hình tham số training
training_args = Seq2SeqTrainingArguments(
    output_dir="./math_ocr_results",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    evaluation_strategy="steps",
    save_steps=1000,
    eval_steps=1000,
    logging_steps=200,
    learning_rate=4e-5,
    num_train_epochs=epochs,
    save_total_limit=2,
    fp16=True, # Bật Mixed Precision cho GPU
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    report_to="none" # Tắt wandb nếu không dùng
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=image_processor, # Trick: Truyền image_processor vào đây để trainer biết cách pad ảnh nếu cần
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["test"],
    data_collator=default_data_collator,
)

print("🚀 Bắt đầu training...")
trainer.train()

# %% [markdown]
# ## 6. Lưu và Tải Model

# %%
save_path = "./antigravity_model_final"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
if MODEL_TYPE == "trocr_finetune":
    processor.save_pretrained(save_path)
else:
    image_processor.save_pretrained(save_path)

print(f"Model saved to {save_path}")

# Nén để download
import shutil
shutil.make_archive('antigravity_model_final', 'zip', save_path)
print("✅ DONE! Hãy tải file 'antigravity_model_final.zip' về.")
