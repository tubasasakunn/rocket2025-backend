from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class HelloResponse(BaseModel):
    message: str

@app.get("/", response_model=HelloResponse)
def hello_world() -> HelloResponse:
    return HelloResponse(message="hello world")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)