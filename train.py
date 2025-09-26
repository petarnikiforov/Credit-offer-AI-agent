import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from transformers import T5Tokenizer
from datasets import Dataset


model_name = "./nbo_model"
checkpoint = "./nbo_model/checkpoint-4500"
CSV_FILE = "./data/train_data2.csv" 

# Base model without any training
# model_name0 = "flan-t5-small" 

# Trained first time with smaller amount of data
# CSV_FILE0 = "./data/train_data.csv" 

# Reading of the csv file with the panda library
df = pd.read_csv(CSV_FILE)

required_cols = ["rfm", "risk", "age", "gender", "income", "active_loans", "offer_text"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column in CSV: {col}")

# Formating inputs and outputs
def format_input(row):
    return (
        f"Customer profile → "
        f"RFM: {row['rfm']}, "
        f"Risk score: {row['risk']}, "
        f"Age: {row['age']}, "
        f"Gender: {row['gender']}, "
        f"Income: {row['income']}, "
        f"Active loans: {row['active_loans']}"
    )

# Structured information for entry data 
inputs = df.apply(format_input, axis=1).tolist()

# Structured data for expected output and to adjust the weight based on the loss 
outputs = df["offer_text"].tolist()

# Creating the dataset to be passed to the model
dataset = Dataset.from_dict({"input_text": inputs, "target_text": outputs})

print("CSV loaded & formatted correctly")

# Setting variables for tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tokenizing the dataset function to pass it to the model
def tokenize_function(batch):
    return tokenizer(batch["input_text"], text_target=batch["target_text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training configuration
training_args = TrainingArguments(
    output_dir="./nbo_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# Save model and Tokenizer
model.save_pretrained("./nbo_model", safe_serialization=False)
tokenizer.save_pretrained("./nbo_model")

print(tokenizer.decode(tokenized_dataset[0]["labels"], skip_special_tokens=True))

print("✅ Training finished. Model saved in ./nbo_model")
