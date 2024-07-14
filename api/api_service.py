from typing import List
from fastapi import FastAPI, Request
from pydantic import BaseModel
import time
import logging
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, Column, Integer, String, Float, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from databases import Database
from starlette.middleware.base import BaseHTTPMiddleware

from models.RAG import RAG_pipeline
from models.rating_predict import predict_with_model

DATABASE_URL = "mysql+aiomysql://root:password@localhost/fastapi_metrics"
# my password is password only

# setting up SQLAlchemy
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
origins = [
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestMetrics(Base):
    __tablename__ = "request_metrics"

    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String(255), index=True)
    response_time = Column(Float)
    timestamp = Column(TIMESTAMP)
    

Base.metadata.create_all(bind=engine)




request_id_counter = 0
request_count = 0
request_start_time = time.time()
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    request_id: int
    results: List[str]

class ReviewRequest(BaseModel):
    review: str

class RatingResponse(BaseModel):
    request_id: int
    rating: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect() 
    try:
        yield  
    finally:
        await database.disconnect()
        

app = FastAPI(lifespan=lifespan)
database = Database(DATABASE_URL)

# Middleware to measure response time and log to database
class MeasureResponseTimeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        async with database.transaction():
            query = RequestMetrics.__table__.insert().values(
                endpoint=request.url.path,
                response_time=process_time
            )
            await database.execute(query)

        logging.info(f"Response time for {request.url.path}: {process_time} seconds")
        return response

app.add_middleware(MeasureResponseTimeMiddleware)

# Background task to log requests per second
async def log_rps():
    global request_count, request_start_time
    while True:
        await asyncio.sleep(60)  # Log every 60 seconds
        elapsed_time = time.time() - request_start_time
        rps = request_count / elapsed_time
        logging.info(f"Requests per second: {rps}")
        request_count = 0
        request_start_time = time.time()

async def lifespan(app: FastAPI):
    # Startup code
    asyncio.create_task(log_rps())
    yield
    # Shutdown code

app.router.lifespan = lifespan
    

@app.post("/predict_rating", response_model=RatingResponse)
async def predict_rating(request: ReviewRequest):
    global request_id_counter, request_count
    request_id_counter += 1
    request_count += 1

    rating = predict_with_model(request.review)
    return RatingResponse(request_id=request_id_counter, rating=rating)

@app.post("/get_answer", response_model=QueryResponse)
async def get_answer(request: QueryRequest):
    global request_id_counter, request_count
    request_id_counter += 1
    request_count += 1

    results = RAG_pipeline(request.query)
    return QueryResponse(request_id=request_id_counter, results=results)

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
