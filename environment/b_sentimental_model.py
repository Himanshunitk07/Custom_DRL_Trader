# b_sentimental_model.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from concurrent.futures import ThreadPoolExecutor

# Load model and tokenizer once
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
model.eval()

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    return {
        "negative": probs[0].item(),
        "neutral": probs[1].item(),
        "positive": probs[2].item(),
        "score": probs[2].item() - probs[0].item()  # sentiment score
    }

def analyze_sentiment_batch(texts, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(analyze_sentiment, texts))
