from fastapi import FastAPI
from mlfunctions import Predictor
from pydantic import BaseModel
import uvicorn

pred = Predictor()
app = FastAPI()


class PredictItem(BaseModel):
    species: str
    symptoms: list


class TrainingData(BaseModel):
    species: str
    symptoms: list
    disease: str


@app.get("/")
def root():
    return {"Ai tonveto.com"}


# GET a list of symptoms
# Example: GET localhost/symptoms/chat
@app.get("/symptoms/{species}")
def symptoms(species: str):
    relevant_symptoms = []

    if species.lower() in ['chat', 'chien']:
        relevant_symptoms = pred.get_symptoms()[0]
    elif species.lower() in ['volaille', 'betail', 'vache', 'cheval', 'chevre', 'mouton', 'poule', 'dinde', 'canard', 'cochon', 'ane']:
        relevant_symptoms = pred.get_symptoms()[1]

    return {"result": relevant_symptoms}


# Will predict the disease by providing animal species and symptoms.
# POST data in the format specified by predictItem to this endpoint to get symptoms
# Example: POST localhost/predict
# Data to be POST'ed:
# {
#    "species":"chien",
#    "symptoms":[
#       "Attacking other animals",
#       "humans and even inanimate objects",
#       "Licking",
#       "biting",
#       "chewing at the bite site",
#       "Fever",
#       "Hypersensitivity",
#       "Hiding in dark places"
#    ]
# }
@app.post("/predict")
def predictFromPost(data: PredictItem):
    predictdata = {'species': data.species, 'symptoms': data.symptoms}
    # print(predictdata)
    # print(pred.predict_disease(predictdata))

    return {"result": pred.predict_disease(predictdata)}


# Will update the datasets and support model retraining.
# Data to be POST'ed:
# {
#    "species":"chien",
#    "symptoms":[
#       "Attacking other animals",
#       "humans and even inanimate objects",
#       "Licking",
#       "biting",
#       "chewing at the bite site",
#       "Fever",
#       "Hypersensitivity",
#       "Hiding in dark places"
#    ],
#   "disease": "cancer"
# }

@app.post("/update")
def update(data: TrainingData):
    update_input = {'species': data.species,
                    'symptoms': data.symptoms, 'disease': data.disease}
    # print(update_input)

    pred.update_dataset(update_input)
    pred.triggerTrain()

    return {"result": "updated dataset + retrain model"}


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)