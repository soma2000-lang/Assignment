## Assesment

FastAPI weather service serving JSON and XML outputs

## Installation and Running

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env file with your https://rapidapi.com/weatherapi/api/weatherapi-com API key
python -m api
```

Data Requirements: Transformers often require large amounts of data for effective training. Smaller datasets can lead to overfitting or suboptimal performance. Fine-Tuning Challenges: Fine-tuning large pre-trained models for specific tasks can be difficult and might require a substantial amount of task-specific data.: Fine-tuning large pre-trained models for specific tasks can be difficult and might require a substantial amount of task-specific data.Fixed Sequence Length: Transformers rely on fixed-size input sequences due to their positional embeddings. Handling variable-length inputs efficiently can be a challenge.Lack of Causality in Standard Attention: The standard self-attention mechanism used in Transformers doesn’t inherently capture causality. In applications like autoregressive language modeling, modifications like masked attention are needed.