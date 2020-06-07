from fastapi import FastAPI

app = FastAPI()


@app.get("/nlp/foucault")
async def root():
    return {"bert": "Hello Foucault"}


@app.get("/nlp/Gyyatri")
async def root():
    return {"message": "Hello Gyyatri"}