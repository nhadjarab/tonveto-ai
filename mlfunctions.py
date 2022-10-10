import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')


class Predictor:
    def __init__(self):
        print("Initial Training - Please wait ...")
        self.model_catdog = self.train_model_catdog()
        self.model_poultryLivestock = self.train_model_poultryLivestock()
        print("Initial Training - Done!")

    def datasets(self):
        catdog_dataset = pd.read_excel('./datasets/cat_dog_dataset.xlsx')
        poultryLivestock_dataset = pd.read_excel(
            './datasets/poultry_livestock_dataset.xlsx')
        return catdog_dataset, poultryLivestock_dataset

    def training_data(self, all_symptoms, symptoms_per_disease):
        input_array = np.zeros(
            (len(symptoms_per_disease), len(all_symptoms)), dtype=int).tolist()
        for i in range(len(symptoms_per_disease)):
            for j in range(len(symptoms_per_disease[i])):
                for m in range(len(all_symptoms)):
                    if symptoms_per_disease[i][j].__contains__(all_symptoms[m]):
                        input_array[i][m] = 1

        output_array = list(np.arange(len(symptoms_per_disease)))

        return input_array, output_array

    def get_symptoms_catdog(self):

        catdog_dataset, poultryLivestock_dataset = self.datasets()
        species_catdog = catdog_dataset['Species'].tolist()
        symptoms_catdog = catdog_dataset['Symptoms'].tolist()

        symptoms_per_disease_catdog_splitted = []
        for i in range(len(symptoms_catdog)):
            symptoms_per_disease_catdog_splitted.append(
                (symptoms_catdog[i].lower().split(',')))
        symptoms_per_disease_catdog = []
        for i in range(len(symptoms_per_disease_catdog_splitted)):
            array = []
            for j in range(len(symptoms_per_disease_catdog_splitted[i])):
                array.append(
                    " ".join(symptoms_per_disease_catdog_splitted[i][j].split()))
            symptoms_per_disease_catdog.append(array)

        # get all possible symptoms
        all_symptoms_catdog = []
        for i in range(len(symptoms_per_disease_catdog)):
            for j in range(len(symptoms_per_disease_catdog[i])):
                symptom = symptoms_per_disease_catdog[i][j].lower()
                all_symptoms_catdog.append(" ".join(symptom.split()))
        all_symptoms_catdog_list = list(set(all_symptoms_catdog))
        all_symptoms_catdog_list.sort()

        return all_symptoms_catdog_list, symptoms_per_disease_catdog

    def get_symptoms_poultryLivestock(self):

        catdog_dataset, poultryLivestock_dataset = self.datasets()
        species_poultryLivestock = poultryLivestock_dataset['Species'].tolist()
        symptoms_poultryLivestock = poultryLivestock_dataset['Symptoms'].tolist(
        )
        symptoms_per_disease_poultryLivestock_splitted = []
        for i in range(len(symptoms_poultryLivestock)):
            symptoms_per_disease_poultryLivestock_splitted.append(
                (symptoms_poultryLivestock[i].lower().split(',')))
        symptoms_per_disease_poultryLivestock = []
        for i in range(len(symptoms_per_disease_poultryLivestock_splitted)):
            array = []
            for j in range(len(symptoms_per_disease_poultryLivestock_splitted[i])):
                array.append(
                    " ".join(symptoms_per_disease_poultryLivestock_splitted[i][j].split()))
            symptoms_per_disease_poultryLivestock.append(array)

        all_symptoms_poultryLivestock = []
        for i in range(len(symptoms_per_disease_poultryLivestock)):
            for j in range(len(symptoms_per_disease_poultryLivestock[i])):
                symptom = symptoms_per_disease_poultryLivestock[i][j].lower()
                all_symptoms_poultryLivestock.append(" ".join(symptom.split()))
        all_symptoms_poultryLivestock_list = list(
            set(all_symptoms_poultryLivestock))
        all_symptoms_poultryLivestock_list.sort()

        return all_symptoms_poultryLivestock_list, symptoms_per_disease_poultryLivestock

    def train_model_catdog(self):

        all_symptoms_catdog_list, symptoms_per_disease_catdog = self.get_symptoms_catdog()

        x_catdog, y_catdog = self.training_data(
            all_symptoms_catdog_list, symptoms_per_disease_catdog)

        x_train = torch.tensor(x_catdog, dtype=torch.float)
        y_train = torch.tensor(y_catdog, dtype=torch.long)

        layers = [
            torch.nn.Linear(x_train.shape[1], 10),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(10, 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(4, len(y_train)),
        ]
        model = torch.nn.Sequential(*layers)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss_form = torch.nn.CrossEntropyLoss()
        model.train()

        accuracy_train = []

        for epoch in range(1000):
            optimizer.zero_grad()
            y_pred = model(x_train).squeeze(-1)
            loss = loss_form(y_pred, y_train)
            loss.backward()
            optimizer.step()
        return model

    def train_model_poultryLivestock(self):

        all_symptoms_poultryLivestock_list, symptoms_per_disease_poultryLivestock = self.get_symptoms_poultryLivestock()

        x_poultryLivestock, y_poultryLivestock = self.training_data(
            all_symptoms_poultryLivestock_list, symptoms_per_disease_poultryLivestock)

        x_train = torch.tensor(x_poultryLivestock, dtype=torch.float)
        y_train = torch.tensor(y_poultryLivestock, dtype=torch.long)

        layers = [
            torch.nn.Linear(x_train.shape[1], 10),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(10, 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(4, len(y_train)),
        ]
        model = torch.nn.Sequential(*layers)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss_form = torch.nn.CrossEntropyLoss()
        model.train()

        accuracy_train = []

        for epoch in range(1000):
            optimizer.zero_grad()
            y_pred = model(x_train).squeeze(-1)
            loss = loss_form(y_pred, y_train)
            loss.backward()
            optimizer.step()

        return model

    def get_symptoms(self):
        symptoms_catDog, _ = self.get_symptoms_catdog()
        symptoms_poultryLivestock, _ = self.get_symptoms_poultryLivestock()
        return symptoms_catDog, symptoms_poultryLivestock

    def transform_input_catdog(self, input):
        symptoms_catdog = input['symptoms']
        all_symptoms, _ = self.get_symptoms_catdog()
        input_array = [0]*len(all_symptoms)
        for i in range(len(symptoms_catdog)):
            for m in range(len(all_symptoms)):
                if symptoms_catdog[i].lower().__contains__(all_symptoms[m]):
                    input_array[m] = 1
        return input_array

    def transform_input_poultryLivestock(self, input):
        symptoms_poultryLivestock = input['symptoms']
        all_symptoms, _ = self.get_symptoms_poultryLivestock()
        input_array = [0]*len(all_symptoms)
        for i in range(len(symptoms_poultryLivestock)):
            for m in range(len(all_symptoms)):
                if symptoms_poultryLivestock[i].lower().__contains__(all_symptoms[m]):
                    input_array[m] = 1
        return input_array

    def predict_disease_catdog(self, input):
        test_array = self.transform_input_catdog(input)
        catdog_dataset, poultryLivestock_dataset = self.datasets()
        disease_catdog = catdog_dataset['Disease'].tolist()
        indices = torch.topk(self.model_catdog(torch.tensor(
            test_array, dtype=torch.float)).squeeze(-1), 2)[1].tolist()
        return [disease_catdog[indices[0]].lower(), disease_catdog[indices[1]].lower()]

    def predict_disease_poultryLivestock(self, input):
        test_array = self.transform_input_poultryLivestock(input)
        catdog_dataset, poultryLivestock_dataset = self.datasets()
        disease_poultryLivestock = poultryLivestock_dataset['Disease'].tolist()
        indices = torch.topk(self.model_poultryLivestock(torch.tensor(
            test_array, dtype=torch.float)).squeeze(-1), 2)[1].tolist()
        return [disease_poultryLivestock[indices[0]].lower(), disease_poultryLivestock[indices[1]].lower()]

    def predict_disease(self, input):
        if input['species'].lower() in ['chat', 'chien']:
            return self.predict_disease_catdog(input)
        elif input['species'].lower() in ['volaille', 'betail', 'vache', 'cheval', 'chevre', 'mouton', 'poule', 'dinde', 'canard','cochon', 'ane']:
            return self.predict_disease_poultryLivestock(input)

    def triggerTrain(self):
        print("Training - Please wait ...")
        self.model_catdog = self.train_model_catdog()
        self.model_poultryLivestock = self.train_model_poultryLivestock()
        print("Training - Done!")

        return "200"

    def update_catDog_datasets(self, update_input):
        df_test = pd.DataFrame([[update_input['species'], update_input['disease'], ",".join(
            update_input['symptoms'])]], columns=['Species', 'Disease', 'Symptoms'])
        cat_dog_dataset = pd.read_excel('./datasets/cat_dog_dataset.xlsx')
        cat_dog_dataset = pd.concat(
            [cat_dog_dataset, df_test], ignore_index=True)
        writer = pd.ExcelWriter('./datasets/cat_dog_dataset.xlsx')
        cat_dog_dataset.to_excel(writer, index=False)
        writer.save()

    def update_poultryLivestock_datasets(self, update_input):
        df_test = pd.DataFrame([[update_input['species'], update_input['disease'], ",".join(
            update_input['symptoms'])]], columns=['Species', 'Disease', 'Symptoms'])
        poultryLivestock_dataset = pd.read_excel(
            './datasets/poultry_livestock_dataset.xlsx')
        poultryLivestock_dataset = pd.concat(
            [poultryLivestock_dataset, df_test], ignore_index=True)
        writer = pd.ExcelWriter('./datasets/poultry_livestock_dataset.xlsx')
        poultryLivestock_dataset.to_excel(writer, index=False)
        writer.save()

    def update_dataset(self, update_input):
        if update_input['species'].lower() in ['chat', 'chien']:
            self.update_catDog_datasets(update_input)
        elif update_input['species'].lower() in ['volaille', 'betail', 'vache', 'cheval', 'chevre', 'mouton', 'poule', 'dinde', 'canard', 'cochon', 'ane']:
            self.update_poultryLivestock_datasets(update_input)
