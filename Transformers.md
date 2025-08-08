
## Introduction

### Key Features

- **Pre-trained Models**: Access to thousands of models from the Hugging Face Hub
- **Multi-framework Support**: Works with PyTorch, TensorFlow, and JAX
- **Easy-to-use Pipelines**: High-level API for common NLP tasks
- **Fine-tuning Capabilities**: Tools for adapting models to specific tasks
- **Community-driven**: Large ecosystem with community contributions

> [!tip] Quick Start If you're new to transformers, start with [[#Pipelines]] for the easiest introduction to the library.

---

## Installation

### Basic Installation

```bash
# Install with pip
pip install transformers

# Install with conda
conda install -c huggingface transformers
```

### Framework-specific Installation

```bash
# For PyTorch users
pip install transformers[torch]

# For TensorFlow users
pip install transformers[tf-cpu]  # CPU version
pip install transformers[tf]      # GPU version

# For development
pip install transformers[dev]
```

### Additional Dependencies

```bash
# For audio processing
pip install transformers[audio]

# For vision tasks
pip install transformers[vision]

# For speech processing
pip install transformers[speech]

# Install all optional dependencies
pip install transformers[all]
```

> [!warning] GPU Requirements For GPU acceleration, ensure you have CUDA installed and compatible PyTorch/TensorFlow versions.

---

## Core Concepts

### 1. Tokenizers

#tokenizers #preprocessing

Tokenizers convert text into tokens that models can understand.

```python
from transformers import AutoTokenizer

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize text
text = "Hello, how are you today?"
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
# Output: ['hello', ',', 'how', 'are', 'you', 'today', '?']

# Encode text (tokenize + convert to IDs)
input_ids = tokenizer.encode(text)
print(f"Input IDs: {input_ids}")
# Output: [101, 7592, 1010, 2129, 2024, 2017, 2651, 1029, 102]

# Decode IDs back to text
decoded_text = tokenizer.decode(input_ids)
print(f"Decoded: {decoded_text}")
# Output: [CLS] hello, how are you today? [SEP]
```

> [!note] Special Tokens BERT uses `[CLS]` (classification) and `[SEP]` (separator) tokens. Different models use different special tokens.

### 2. Models

#models #inference

Models perform the actual computations and predictions.

```python
from transformers import AutoModel, AutoModelForSequenceClassification
import torch

# Load a base model
model = AutoModel.from_pretrained("bert-base-uncased")

# Load a model for specific task
classifier = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=2
)

# Basic forward pass
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

print(f"Last hidden states shape: {outputs.last_hidden_state.shape}")
print(f"Pooler output shape: {outputs.pooler_output.shape}")
```

### 3. Configuration

#configuration #model-settings

Models have configurations that define their architecture and behavior.

```python
from transformers import AutoConfig

# Load configuration
config = AutoConfig.from_pretrained("bert-base-uncased")
print(f"Hidden size: {config.hidden_size}")
print(f"Number of layers: {config.num_hidden_layers}")
print(f"Number of attention heads: {config.num_attention_heads}")

# Create custom configuration
custom_config = AutoConfig.from_pretrained(
    "bert-base-uncased",
    num_labels=5,
    hidden_dropout_prob=0.2
)
```

---

## Pipelines

#pipelines #high-level-api

> [!success] Easiest Way to Get Started Pipelines provide a high-level, easy-to-use interface for common NLP tasks without requiring deep knowledge of the underlying models.

### 1. Text Classification

#sentiment-analysis #classification

```python
from transformers import pipeline

# Create a sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# Classify single text
result = classifier("I love this movie!")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]

# Classify multiple texts
texts = [
    "I hate this product",
    "This is amazing!",
    "It's okay, not great"
]
results = classifier(texts)
for text, result in zip(texts, results):
    print(f"'{text}' -> {result}")
```

### 2. Named Entity Recognition (NER)

#ner #entity-extraction

```python
# NER pipeline
ner_pipeline = pipeline("ner", aggregation_strategy="simple")

text = "My name is John Smith and I work at Microsoft in Seattle."
entities = ner_pipeline(text)

for entity in entities:
    print(f"Entity: {entity['word']}, Label: {entity['entity_group']}, Score: {entity['score']:.4f}")
```

### 3. Question Answering

#qa #question-answering

```python
# Question answering pipeline
qa_pipeline = pipeline("question-answering")

context = """
The Amazon rainforest is a moist broadleaf forest that covers most of the Amazon basin of South America. 
This basin encompasses 7,000,000 square kilometers, of which 5,500,000 square kilometers are covered by the rainforest.
"""

question = "How large is the Amazon rainforest?"
answer = qa_pipeline(question=question, context=context)

print(f"Question: {question}")
print(f"Answer: {answer['answer']}")
print(f"Score: {answer['score']:.4f}")
```

### 4. Text Generation

#text-generation #gpt

```python
# Text generation pipeline
generator = pipeline("text-generation", model="gpt2")

prompt = "The future of artificial intelligence is"
generated = generator(
    prompt,
    max_length=100,
    num_return_sequences=2,
    temperature=0.7,
    do_sample=True
)

for i, gen in enumerate(generated):
    print(f"Generation {i+1}: {gen['generated_text']}")
```

### 5. Summarization

#summarization #text-summarization

```python
# Summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
"""

summary = summarizer(article, max_length=50, min_length=10, do_sample=False)
print(f"Summary: {summary[0]['summary_text']}")
```

### 6. Translation

#translation #multilingual

```python
# Translation pipeline
translator = pipeline("translation_en_to_fr", model="t5-base")

text = "How are you today?"
translation = translator(text)
print(f"English: {text}")
print(f"French: {translation[0]['translation_text']}")
```

### 7. Fill Mask

#fill-mask #masked-language-modeling

```python
# Fill mask pipeline
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

text = "The capital of France is [MASK]."
predictions = fill_mask(text)

for pred in predictions:
    print(f"Token: {pred['token_str']}, Score: {pred['score']:.4f}")
```

### Custom Pipeline Configuration

#custom-pipeline #advanced-configuration

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Load specific model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create custom pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)

# Use the pipeline
result = sentiment_pipeline("I love programming with transformers!")
print(result)
```

> [!tip] Pipeline Performance For better performance, specify the device parameter and consider using smaller, task-specific models for production.

---

## Models and Tokenizers

#model-architectures #deep-dive

### Working with Different Model Types

```python
from transformers import (
    BertTokenizer, BertModel,
    GPT2Tokenizer, GPT2LMHeadModel,
    T5Tokenizer, T5ForConditionalGeneration,
    RobertaTokenizer, RobertaModel
)

# BERT - Bidirectional Encoder
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# GPT-2 - Autoregressive Language Model
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# T5 - Text-to-Text Transfer Transformer
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")

# RoBERTa - Robustly Optimized BERT
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base")
```

### Model Usage Examples

#### BERT for Feature Extraction

#bert #feature-extraction

```python
import torch

text = "Transformers are powerful models for NLP."
inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = bert_model(**inputs)
    
# Get embeddings
last_hidden_states = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]
pooled_output = outputs.pooler_output  # Shape: [batch_size, hidden_size]

print(f"Last hidden states shape: {last_hidden_states.shape}")
print(f"Pooled output shape: {pooled_output.shape}")
```

#### GPT-2 for Text Generation

#gpt2 #text-generation

```python
def generate_text(prompt, model, tokenizer, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

prompt = "The benefits of machine learning include"
generated = generate_text(prompt, gpt2_model, gpt2_tokenizer)
print(f"Generated text: {generated}")
```

#### T5 for Text-to-Text Tasks

#t5 #text-to-text

```python
# Text summarization with T5
def summarize_with_t5(text, model, tokenizer):
    input_text = f"summarize: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        summary_ids = model.generate(
            inputs.input_ids,
            max_length=150,
            min_length=50,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

text_to_summarize = """
Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task. It uses algorithms to identify patterns in data and make predictions or decisions based on those patterns.
"""

summary = summarize_with_t5(text_to_summarize, t5_model, t5_tokenizer)
print(f"Summary: {summary}")
```

### Tokenizer Advanced Features

#tokenizer-advanced #preprocessing

```python
# Advanced tokenization options
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hello world! How are you?"

# Basic tokenization
basic_encoding = tokenizer(text)
print(f"Basic encoding: {basic_encoding}")

# With padding and truncation
padded_encoding = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=20,
    return_tensors="pt"
)
print(f"Padded encoding shape: {padded_encoding['input_ids'].shape}")

# Batch processing
texts = ["Hello world!", "How are you?", "I'm fine, thanks!"]
batch_encoding = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)
print(f"Batch encoding shape: {batch_encoding['input_ids'].shape}")

# Get attention masks
print(f"Attention masks: {batch_encoding['attention_mask']}")
```

---

## Training and Fine-tuning

#training #fine-tuning #machine-learning

> [!important] Training Requirements Fine-tuning requires significant computational resources. Consider using Google Colab Pro or similar cloud platforms for GPU access.

### 1. Fine-tuning for Classification

#classification #supervised-learning

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from torch.utils.data import Dataset
import torch

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Example data
train_texts = ["I love this!", "This is terrible", "Pretty good", "Not bad"]
train_labels = [1, 0, 1, 1]  # 1 for positive, 0 for negative

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2
)

# Create dataset
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    load_best_model_at_end=True,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./fine-tuned-model")
```

### 2. Fine-tuning for Question Answering

#question-answering #fine-tuning

```python
from transformers import AutoModelForQuestionAnswering

# Load QA model
qa_model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        context = item['context']
        answer_text = item['answer']
        
        # Find answer start position
        answer_start = context.find(answer_text)
        answer_end = answer_start + len(answer_text)
        
        # Tokenize
        encoding = self.tokenizer(
            question,
            context,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'start_positions': torch.tensor([answer_start], dtype=torch.long),
            'end_positions': torch.tensor([answer_end], dtype=torch.long)
        }

# Example QA data
qa_data = [
    {
        "question": "What is the capital of France?",
        "context": "France is a country in Europe. The capital of France is Paris.",
        "answer": "Paris"
    }
]

qa_dataset = QADataset(qa_data, tokenizer)
```

### 3. Custom Training Loop

#custom-training #pytorch

```python
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

def train_model_custom_loop(model, train_dataset, num_epochs=3, batch_size=16, lr=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')
```

---

## Advanced Usage

#advanced #optimization #performance

### 1. Model Parallelism and Optimization

#parallelism #optimization #gpu

```python
from transformers import AutoModel
import torch.nn as nn

# Load large model with device mapping
large_model = AutoModel.from_pretrained(
    "microsoft/DialoGPT-large",
    device_map="auto",  # Automatically distribute across GPUs
    torch_dtype=torch.float16,  # Use half precision
    low_cpu_mem_usage=True
)

# 8-bit quantization (requires bitsandbytes)
try:
    from transformers import BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    
    quantized_model = AutoModel.from_pretrained(
        "facebook/opt-6.7b",
        quantization_config=quantization_config,
        device_map="auto"
    )
except ImportError:
    print("bitsandbytes not installed. Skipping quantization example.")
```

### 2. Gradient Checkpointing

#gradient-checkpointing #memory-optimization

```python
# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Training with gradient checkpointing
def train_with_checkpointing(model, dataloader, optimizer):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Forward pass with gradient checkpointing
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
```

### 3. Custom Model Architectures

#custom-models #architecture

```python
from transformers import BertModel, BertConfig
import torch.nn as nn

class CustomBertClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

# Usage
custom_model = CustomBertClassifier("bert-base-uncased", num_classes=3)
```

### 4. Working with Hugging Face Hub

#huggingface-hub #model-sharing

```python
from huggingface_hub import HfApi, Repository
from transformers import AutoModel, AutoTokenizer

# Load model from Hub
def load_model_from_hub(repo_name):
    model = AutoModel.from_pretrained(repo_name)
    tokenizer = AutoTokenizer.from_pretrained(repo_name)
    return model, tokenizer

# Search models on Hub
from huggingface_hub import list_models

# List all BERT models
bert_models = list_models(filter="bert")
for model in list(bert_models)[:5]:  # Show first 5
    print(f"Model: {model.modelId}, Downloads: {model.downloads}")
```

---

## Best Practices

#best-practices #guidelines #performance

> [!success] Key Recommendations Following these best practices will help you avoid common pitfalls and optimize your transformer usage.

### 1. Memory Management

#memory-management #optimization

```python
import gc
import torch

def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Use context managers for inference
@torch.no_grad()
def efficient_inference(model, inputs):
    outputs = model(**inputs)
    return outputs

# Batch processing for large datasets
def process_large_dataset(model, tokenizer, texts, batch_size=32):
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Process batch
        with torch.no_grad():
            outputs = model(**inputs)
            results.extend(outputs.logits.cpu().numpy())
        
        # Clear memory periodically
        if i % (batch_size * 10) == 0:
            clear_memory()
    
    return results
```

### 2. Error Handling and Logging

#error-handling #logging

```python
import logging
from transformers import logging as transformers_logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure transformers logging
transformers_logging.set_verbosity_info()

def safe_model_loading(model_name):
    try:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Successfully loaded {model_name}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {str(e)}")
        return None, None

def safe_inference(model, tokenizer, text):
    try:
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        return outputs
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        return None
```

### 3. Configuration Management

#configuration #settings

```python
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class ModelConfig:
    model_name: str
    max_length: int = 512
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    output_dir: str = "./results"
    
    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, json_path: str):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

# Usage
config = ModelConfig(
    model_name="bert-base-uncased",
    max_length=256,
    batch_size=16
)
```

### 4. Evaluation Metrics

#evaluation #metrics

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Use with Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
```

---

## Troubleshooting

#troubleshooting #debugging #common-issues

> [!bug] Common Issues This section covers the most frequently encountered problems and their solutions.

### 1. CUDA Out of Memory

#cuda #memory-issues

```python
# Solutions:
# Reduce batch size
training_args.per_device_train_batch_size = 8

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use gradient accumulation
training_args.gradient_accumulation_steps = 4

# Use mixed precision training
training_args.fp16 = True
```

### 2. Slow Training

#performance #speed

```python
# Use DataLoader with multiple workers
train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Compile model (PyTorch 2.0+)
compiled_model = torch.compile(model)
```

### 3. Model Not Converging

#convergence #training-issues

```python
# Adjust learning rate
training_args.learning_rate = 5e-5

# Use learning rate scheduler
training_args.lr_scheduler_type = "cosine"
training_args.warmup_ratio = 0.1

# Increase training epochs
training_args.num_train_epochs = 5
```

### 4. Tokenization Issues

#tokenization #preprocessing-issues

```python
# Handle special tokens
tokenizer.add_special_tokens({
    'additional_special_tokens': ['<CUSTOM_TOKEN>']
})
model.resize_token_embeddings(len(tokenizer))

# Handle long sequences
def handle_long_text(text, tokenizer, max_length=512, stride=256):
    tokens = tokenizer.tokenize(text)
    chunks = []
    
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
    
    return chunks
```


## 📚 Additional Resources

### Official Documentation

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [Hugging Face Hub](https://huggingface.co/models)
- [Transformers Course](https://huggingface.co/course)

### Tutorials and Examples

- [Fine-tuning Tutorial](https://huggingface.co/docs/transformers/training)
- [Custom Datasets Guide](https://huggingface.co/docs/datasets)
- [Model Optimization](https://huggingface.co/docs/transformers/performance)

### Community Resources

- [Hugging Face Forums](https://discuss.huggingface.co/)
- [GitHub Repository](https://github.com/huggingface/transformers)
- [Discord Community](https://discord.com/invite/JfAtkvEtRb)

---

## 🏷️ Tags

#transformers #nlp #machine-learning #python #huggingface #bert #gpt #t5 #pytorch #tensorflow #fine-tuning #pipelines #tokenization #deep-learning #ai #documentation

---

## 📝 Notes

> [!note] Last Updated This documentation covers Transformers library version 4.x. Check the official documentation for the latest updates and features.

> [!warning] GPU Memory Large models require significant GPU memory. Monitor your usage and consider using model parallelism or quantization for very large models.

> [!tip] Getting Started If you're new to transformers, start with the [[#Pipelines]] section for quick wins, then move to [[#Models and Tokenizers]] for more control.

---

## 🔍 Quick Search Tags

**By Task:**

- #sentiment-analysis #classification #ner #question-answering #summarization #translation #text-generation

**By Model:**

- #bert #gpt2 #t5 #roberta #distilbert #electra

**By Technique:**

- #fine-tuning #transfer-learning #feature-extraction #embeddings

**By Level:**

- #beginner #intermediate #advanced

**By Framework:**

- #pytorch #tensorflow #jax

**By Resource:**

- #memory-optimization #gpu #cpu #performance

---

## 📋 Quick Reference Commands

### Installation Commands

```bash
# Basic installation
pip install transformers

# With PyTorch
pip install transformers[torch]

# With TensorFlow
pip install transformers[tf]

# Development version
pip install git+https://github.com/huggingface/transformers.git
```

### Essential Imports

```python
# Core imports
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    pipeline,
    Trainer,
    TrainingArguments
)

# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader

# Utilities
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
```

### Common Pipeline Tasks

```python
# Available pipeline tasks
PIPELINE_TASKS = [
    "sentiment-analysis",
    "ner",
    "question-answering",
    "summarization",
    "translation",
    "text-generation",
    "fill-mask",
    "feature-extraction",
    "text-classification",
    "token-classification"
]
```

### Model Loading Patterns

```python
# Pattern 1: Auto classes (recommended)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Pattern 2: Specific classes
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Pattern 3: With configuration
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, config=config)
```

---

## 🧪 Experiments and Tests

### Performance Benchmarks

Create linked notes for:

- [[Model Speed Comparisons]]
- [[Memory Usage Analysis]]
- [[Accuracy Benchmarks]]

### Custom Implementations

Track your experiments:

- [[Custom BERT Fine-tuning Results]]
- [[Multi-label Classification Experiments]]
- [[Domain Adaptation Studies]]

---

## 🔧 Development Setup

### Environment Configuration

```yaml
# environment.yml
name: transformers-env
dependencies:
  - python=3.9
  - pytorch
  - torchvision
  - torchaudio
  - pip
  - pip:
    - transformers
    - datasets
    - tokenizers
    - accelerate
    - evaluate
    - wandb
```

### VS Code Settings

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "files.associations": {
        "*.md": "markdown"
    }
}
```

---

## 🎯 Project Templates

### Classification Project Structure

```
project/
├── data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── models/
│   └── fine_tuned/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── inference.py
├── config/
│   └── training_config.json
└── requirements.txt
```

### Training Script Template

```python
#!/usr/bin/env python3
"""
Training script template for transformers fine-tuning
Usage: python train.py --config config/training_config.json
"""

import argparse
import json
import logging
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = json.load(f)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Your training code here
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

if __name__ == "__main__":
    main()
```

---

## 📊 Model Comparison Table

|Model|Parameters|Use Case|Speed|Memory|Accuracy|
|---|---|---|---|---|---|
|DistilBERT|66M|Fast classification|⭐⭐⭐⭐⭐|⭐⭐⭐⭐|⭐⭐⭐|
|BERT-base|110M|General NLP|⭐⭐⭐|⭐⭐⭐|⭐⭐⭐⭐|
|BERT-large|340M|High accuracy|⭐⭐|⭐⭐|⭐⭐⭐⭐⭐|
|RoBERTa|125M|Robust performance|⭐⭐⭐|⭐⭐⭐|⭐⭐⭐⭐⭐|
|GPT-2|117M-1.5B|Text generation|⭐⭐⭐|⭐⭐|⭐⭐⭐⭐|
|T5-base|220M|Text-to-text|⭐⭐|⭐⭐|⭐⭐⭐⭐|

---

## 🚀 Production Deployment

### Model Serving with FastAPI

```python
from fastapi import FastAPI
from transformers import pipeline
import uvicorn

app = FastAPI()

# Load model once at startup
classifier = pipeline("sentiment-analysis")

@app.post("/predict")
async def predict(text: str):
    result = classifier(text)
    return {"prediction": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Docker Configuration

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
```

---

## 💡 Tips and Tricks

### Memory Optimization

- Use `torch.no_grad()` for inference
- Clear cache with `torch.cuda.empty_cache()`
- Use gradient checkpointing for large models
- Consider model quantization

### Training Optimization

- Use mixed precision training (`fp16=True`)
- Implement gradient accumulation
- Use learning rate schedulers
- Monitor with wandb or tensorboard

### Debugging Tips

- Check tokenizer output shapes
- Verify label encoding
- Monitor loss curves
- Use smaller datasets for initial testing

---

## ⚡ Code Snippets Library

### Quick Model Testing

```python
def quick_test(model_name, text):
    """Quick test function for any model"""
    pipe = pipeline("sentiment-analysis", model=model_name)
    return pipe(text)

# Usage
result = quick_test("distilbert-base-uncased-finetuned-sst-2-english", "I love this!")
```

### Batch Processing

```python
def batch_process(texts, model, tokenizer, batch_size=32):
    """Process texts in batches"""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        results.extend(outputs.logits.cpu().numpy())
    return results
```

### Model Comparison

```python
def compare_models(text, model_names):
    """Compare multiple models on the same text"""
    results = {}
    for name in model_names:
        pipe = pipeline("sentiment-analysis", model=name)
        results[name] = pipe(text)
    return results
```

---

## 📅 Version History

- **v1.0** - Initial documentation creation
- **v1.1** - Added advanced usage examples
- **v1.2** - Enhanced troubleshooting section
- **v1.3** - Added production deployment examples
- **v1.4** - Obsidian formatting and internal links

---

_Created: [[2024-08-07]]_ _Last Modified: [[2024-08-07]]_ _Status: Complete_ _Reviewed: ✅_
