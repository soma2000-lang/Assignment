from fastapi import APIRouter, Response
from app.models.request_models import 
from app.services.weather_service import get_weather_by_city

router = APIRouter()



@app.post("/predict_rating", response_model=RatingResponse)
async def predict_rating(request: ReviewRequest):
    global request_id_counter
    request_id_counter += 1
    # Dummy implementation for rating prediction
    rating = len(request.review) % 5 + 1
    return RatingResponse(request_id=request_id_counter, rating=rating)

@app.post("/get_answer", response_model=QueryResponse)
async def get_answer(request: QueryRequest):
    global request_id_counter
    request_id_counter += 1
    results = RAG_pipeline(request.query, corpus, get_embeddings, bm25, nonot5, sides, summarize_documents)
    return QueryResponse(request_id=request_id_counter, results=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)