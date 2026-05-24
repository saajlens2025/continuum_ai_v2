from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Continuum AI is alive!"}

@app.get("/health")
def health():
    return {"status": "ok"}
