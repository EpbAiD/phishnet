# Cybersecurity-Focused LLM Options for Phishing Detection

## Current Setup
- **Model**: Qwen2.5-0.5B-Instruct (500M parameters)
- **Approach**: Prompt engineering (no fine-tuning)
- **Strengths**: Fast, lightweight, good instruction following
- **Limitations**: General-purpose model, not specialized in cybersecurity

---

## Better Cybersecurity-Focused LLM Options

### 1. **SecureBERT / CyBERT** ⭐ Recommended
**Hugging Face**: `jackaduma/SecBERT` or `raduion/cybert`

**Why it's better:**
- Pre-trained specifically on cybersecurity texts (CVEs, security advisories, threat reports)
- Understands security terminology naturally
- Smaller model (~110M params) - faster inference
- Better at identifying phishing patterns

**How to use:**
```python
MODEL_ID = "jackaduma/SecBERT"
# or
MODEL_ID = "raduion/cybert"
```

**Pros:**
- ✅ Domain-specific knowledge
- ✅ Faster than general-purpose LLMs
- ✅ Better understanding of security concepts
- ✅ Can be fine-tuned on phishing data

**Cons:**
- ⚠️ Smaller than Qwen (may need more fine-tuning)
- ⚠️ Less conversational (better for classification than explanation)

---

### 2. **SecLLM / CyberLLaMA**
**Hugging Face**: Research models from security labs

**Why it's better:**
- Trained on cybersecurity datasets
- Better at threat intelligence interpretation
- Can explain security concepts in user-friendly language

**Status**:
- 🔍 Emerging research area
- 🔬 Check for latest models on Hugging Face with "cyber" or "security" tags

---

### 3. **Llama-3.2-1B / Llama-3.2-3B** (Instruction-tuned)
**Hugging Face**: `meta-llama/Llama-3.2-1B-Instruct` or `meta-llama/Llama-3.2-3B-Instruct`

**Why it's better:**
- Strong instruction-following abilities
- Better reasoning than smaller models
- Can be fine-tuned on cybersecurity data
- Good balance between size and performance

**Pros:**
- ✅ Excellent prompt engineering support
- ✅ Can understand complex security scenarios
- ✅ Better explanation generation
- ✅ Active community support

**Cons:**
- ⚠️ Larger than Qwen2.5-0.5B (slower inference)
- ⚠️ Requires more RAM (~4-6GB)

---

### 4. **Phi-3-mini / Phi-3.5-mini** (Microsoft)
**Hugging Face**: `microsoft/Phi-3-mini-4k-instruct` or `microsoft/Phi-3.5-mini-instruct`

**Why it's better:**
- High-quality instruction following
- Better reasoning capabilities
- Trained on high-quality data (including technical docs)
- Optimized for efficiency

**Pros:**
- ✅ Excellent for technical explanations
- ✅ Compact size (~3.8B params)
- ✅ Strong safety guardrails
- ✅ Good at simplifying complex concepts

**Cons:**
- ⚠️ Larger than current setup
- ⚠️ May be slower for real-time inference

---

### 5. **DistilBERT / RoBERTa** (Fine-tuned on Security Data)
**Approach**: Take DistilBERT/RoBERTa and fine-tune on phishing explanation dataset

**Why it's better:**
- Smaller, faster models
- Can be specialized for our exact use case
- Better for classification + explanation tasks

**Implementation:**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Fine-tune on our phishing dataset
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
# Fine-tune with our llm_training_dataset.jsonl
```

---

## Recommendation

### For **Prompt Engineering Only** (Current Approach):
**Best Choice: Llama-3.2-1B-Instruct** or **Phi-3-mini**
- Better reasoning and explanation
- Still lightweight enough for CPU inference
- Excellent instruction following

### For **Fine-Tuning** (Future Improvement):
**Best Choice: SecBERT or CyBERT**
- Domain-specific knowledge
- Smaller footprint
- Can be fine-tuned on our phishing explanation dataset
- Faster inference

---

## Implementation Guide

### Option 1: Switch to Llama-3.2-1B (Simple - Just Prompt Engineering)

**Update `src/api/llm_explainer.py`:**
```python
# Change line 23:
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# Keep everything else the same - just better prompt following!
```

**Benefits:**
- ✅ Drop-in replacement
- ✅ Better explanations
- ✅ No fine-tuning needed
- ✅ ~2x better reasoning

**Trade-offs:**
- ⚠️ Slightly slower (but still fast on CPU)
- ⚠️ Uses more RAM (~2-3GB vs 1GB)

---

### Option 2: Fine-Tune SecBERT (Advanced - Best Performance)

**Step 1**: Load SecBERT
```python
MODEL_ID = "jackaduma/SecBERT"
```

**Step 2**: Fine-tune on our phishing dataset
```bash
python src/llm/finetune_explainer.py
```

**Benefits:**
- ✅ **Best accuracy** for phishing explanations
- ✅ Domain-specific knowledge built-in
- ✅ Faster inference than general LLMs
- ✅ Understands security terminology

**Trade-offs:**
- ⚠️ Requires fine-tuning (we already have the pipeline!)
- ⚠️ May need more training data for conversational responses

---

## Testing Recommendations

### Quick Test (5 minutes):
1. Update `MODEL_ID` in `llm_explainer.py`
2. Restart server
3. Test with `/explain` endpoint
4. Compare explanation quality

### Comprehensive Evaluation:
1. Generate 100 test samples
2. Run all models in parallel
3. Measure:
   - Explanation quality (human evaluation)
   - Inference speed
   - RAM usage
   - User-friendliness of language

---

## Conclusion

**For immediate improvement with minimal effort:**
→ Switch to **Llama-3.2-1B-Instruct**

**For best long-term performance:**
→ Fine-tune **SecBERT** on our phishing dataset

**Current setup is fine if:**
- Qwen2.5-0.5B explanations are good enough
- You prioritize speed over accuracy
- Resource constraints are critical

---

## Next Steps

1. **Test Llama-3.2-1B** with current prompt engineering
2. **Evaluate explanation quality** on 50 sample URLs
3. **If needed**, fine-tune SecBERT for domain specialization
4. **Compare all approaches** and pick the best one

Would you like me to implement any of these options?
