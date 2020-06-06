from fastapi import FastAPI

app = FastAPI()


@app.get("/nlp/bert")
async def root():
    return {"bert": "Hello Foucault"}

@app.get("/nlp/openaigpt")
async def root():
    return {"message": "Hello Gyyatri"}