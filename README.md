## Assesment



## Installation and Running

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 api/api_service.py
python3 api/api_test.py
python3 models/preprocess.py
python3 models/RAG.py
python3 models/rating_predict.py
```

Data Requirements: Transformers often require large amounts of data for effective training. Smaller datasets can lead to overfitting or suboptimal performance.Fixed Sequence Length: Transformers rely on fixed-size input sequences due to their positional embeddings. Handling variable-length inputs efficiently can be a challenge.Lack of Causality in Standard Attention: The standard self-attention mechanism used in Transformers doesn’t inherently capture causality. 