from fastapi import FastAPI
app = FastAPI()


@app.get("/")
def root():
    return {"FastAPI": "Hello World"}

# Will get a list of supported symptoms for an animal species.
@app.get("/symptoms/{species}")
def get_symptoms():
    return {"get_symptoms"}

# Will predict the disease by providing animal species and symptoms.
@app.post("/predict")
def predict():
    return {"predict"}

# Will update the datasets and support model retraining.
@ app.post("/update")
def update():
    return {"update": "update"}


