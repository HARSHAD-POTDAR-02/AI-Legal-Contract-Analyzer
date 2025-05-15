from flask import Flask, request, render_template
import re
from textblob import TextBlob
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

app = Flask(__name__)

# === Configuration ===
CONFIG = {
    "models": {
        "llama_model": "meta-llama/Llama-3.2-1b-instruct",
        "summarization_model": "facebook/bart-large-cnn",
        "tokenizer": "meta-llama/Llama-3.2-1b-instruct"
    },
    "contract_types": ["Legal", "Employment", "Non-Disclosure", "Service Agreement", "Intellectual Property"],
    "clause_types": ["Confidentiality", "Payment Terms", "Termination", "Liability", "Governing Law",
                     "Employment Duties"],
    "thresholds": {
        "flag_confidence": 0.5,
        "clause_confidence": 0.5
    },
    "max_lengths": {
        "contract_type": 512,
        "clause_analysis": 384
    }
}

# === Initialize Models ===
tokenizer = AutoTokenizer.from_pretrained(CONFIG["models"]["tokenizer"])

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

llama_model = AutoModelForSequenceClassification.from_pretrained(CONFIG["models"]["llama_model"]).to(device)
summarizer = pipeline("summarization", model=CONFIG["models"]["summarization_model"],
                      tokenizer=CONFIG["models"]["summarization_model"], device=0 if torch.cuda.is_available() else -1)


# === Helper Functions ===
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    return text


def extract_all_clauses(text):
    clauses = re.split(r'(?<=\n)\s*(?=[A-Z][^a-z]|\d+\.|\bSECTION|\bARTICLE|\bCLAUSE|\b[IVXLCDM]+\.)', text)
    refined_clauses = []
    for clause in clauses:
        if len(clause.split()) > 15:
            sentences = re.split(r'(?<=[.!?])\s+', clause)
            refined_clauses.extend(sentences)
        else:
            refined_clauses.append(clause)
    return [preprocess_text(c) for c in refined_clauses if c.strip()]


def classify_clause(clause):
    if "salary" in clause.lower() or "payment" in clause.lower():
        return "Payment Terms"
    elif "confidential" in clause.lower() or "non-disclosure" in clause.lower():
        return "Confidentiality"
    elif "termination" in clause.lower():
        return "Termination"
    elif "dispute" in clause.lower() or "arbitration" in clause.lower():
        return "Dispute Resolution"
    elif "responsibilities" in clause.lower() or "duties" in clause.lower():
        return "Duties"
    else:
        return "General Information"


def get_clause_flag(clause):
    red_flags = ["without notice", "immediate termination", "sole discretion", "penalties", "non-compete",
                 "no guarantee"]
    green_flags = ["guarantee", "reasonable notice", "mutual agreement", "negotiable", "fair", "clear terms"]

    red_flag_count = sum(1 for flag in red_flags if flag in clause.lower())
    green_flag_count = sum(1 for flag in green_flags if flag in clause.lower())
    sentiment_score = TextBlob(clause).sentiment.polarity

    if red_flag_count > green_flag_count:
        return "Red Flag", sentiment_score
    elif green_flag_count > red_flag_count:
        return "Green Flag", sentiment_score
    else:
        return "Neutral", sentiment_score


def predict_contract_type(contract_text):
    inputs = tokenizer(contract_text, truncation=True, padding="max_length",
                       max_length=CONFIG["max_lengths"]["contract_type"], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = llama_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_idx = probs.argmax()

    return CONFIG["contract_types"][pred_idx]


# === Main Analysis Function ===
def analyze_contract(contract_text):
    contract_text = preprocess_text(contract_text)

    # Summarize first 1024 tokens (or characters) of contract for brevity
    summary = summarizer(contract_text[:1024], max_length=150, min_length=30, do_sample=False)
    summary_text = summary[0]['summary_text']

    # Contract Type Prediction
    contract_type = predict_contract_type(contract_text)

    # Clause Analysis
    clauses = extract_all_clauses(contract_text)
    results = []
    for clause in clauses:
        flag, flag_conf = get_clause_flag(clause)
        clause_type = classify_clause(clause)

        results.append({
            "clause": clause[:300],  # truncate for display, you can adjust
            "type": clause_type,
            "flag": flag,
            "flag_confidence": round(flag_conf, 2)
        })

    return results, summary_text, contract_type


# === Flask Routes ===
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files['contract_file']
        if file:
            contract_text = file.read().decode('utf-8')
            results, summary, contract_type = analyze_contract(contract_text)
            return render_template('index.html', results=results, summary=summary, contract_type=contract_type)
        else:
            return "No file uploaded", 400
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
