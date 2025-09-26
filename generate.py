import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#Model to be used for generation
MODEL_DIR = "./nbo_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)


# Function to predict with predifined values
def generate_offer(rfm, risk, age, gender, name, income, active_loans, max_len=128):
    input_text = f"Customer profile → RFM: {rfm}, Risk: {risk}, Age: {age}, Gender: {gender}, Name: {name}, Income: {income}, Active Loans: {active_loans}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    outputs = model.generate(**inputs, max_length=max_len)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Reading data from csv without offer text
df = pd.read_csv("./data/actual_clients.csv")

# Function to validate if there is a missing column in the csv file
required_cols = ["rfm", "risk", "age", "gender", "name", "income", "active_loans"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column in CSV: {col}")
    
print("reached")
# Generate offer for every row in the the actual clients csv
for i, row in df.iterrows():
    offer = generate_offer(
        row["rfm"],
        row["risk"],
        row["age"],
        row["gender"],
        row["name"], 
        row["income"],
        row["active_loans"], 
    )
    print(f"Client {i+1}: {offer}")

add_new = "y"
while True:
  if add_new == "y":
# Adding manually a new client data through the console 
# If the input is with incorrect format returning error and asking again
    line = input(
        "Enter client data in format:\n"
        "rfm,risk,age,gender(M/F),name,income,active_loans\n> "
    ).strip()

    try:
        rfm, risk, age, gender, name, income, active_loans = line.split(",", 6)
        offer = generate_offer(
            int(rfm),
            int(risk),
            int(age),
            gender.strip().upper(),
            name.strip(),
            int(income),
            int(active_loans),
        )
        print("\n--- Generated Offer for Your Client ---")
        print(offer)
    except Exception as e:
        print("❌ Invalid format. Please try again. Error:", e)
        print("Make sure your input follows the exact format: rfm,risk,age,gender(M/F),name,income,active_loans\n")
    add_new = input("\nDo you want to add a new client manually? (y/n): ").strip().lower()
  else: 
      break