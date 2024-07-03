
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import openai_gpt

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api")
async def read_root():
    return {"message": "Hello World"}

@app.post("/api/gpt-fine-tuned/conclusion")
def gemini(request_body: dict):
    query = request_body.get("query", "")
    return openai_gpt.conclude(query)

# Usage
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
