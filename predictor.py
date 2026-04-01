import pickle

class Predictor:

    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance


    def __init__(self):
        if not self._initialized:
            self.model = None
            self.scaler = None
            self._loaded = False
            self._initialized = True


    def _load(self):
        if self._loaded:
            return

        try:
            with open('model/diabetes_model.pkl', 'rb') as file:
                self.model = pickle.load(file)

            with open('model/scaler.pkl', 'rb') as file:
                self.scaler = pickle.load(file)

            self._loaded = True
        except FileNotFoundError:
            self._loaded = False
            self.model = None
            self.scaler = None
            print("ERROR: Model has not been created. Please run notebook/Model.ipynb first.")


    def predict_w_params(self, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age):
        """
        param
        pregnancies : int - Number of pregnancies
        glucose : float - Glucose level
        blood_pressure : float - Blood pressure (mm Hg)
        skin_thickness : float - Skin thickness (mm)
        insulin : float - Insulin level (mu U/ml)
        bmi : float - Body Mass Index
        diabetes_pedigree : float - Diabetes pedigree function
        age : int - Age in years

        returns
        prediction : int - 0 (No diabetes) or 1 (Diabetes)
        probability : float - Probability of having diabetes (0-1)
        """

        if not self._loaded:
            self._load()

        import numpy as np
        patient_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi,
                                  diabetes_pedigree, age]])
        return self.predict_w_patient_data(patient_data)


    def is_ready(self):
        "Check if model is loaded"
        return self._loaded


    def predict_w_patient_data(self, patient_data):
        """
        params
        patient_data : numpy array - Shape (n_samples, 8) with features in correct order

        returns
        prediction : int - 0 (No diabetes) or 1 (Diabetes)
        probability : float - Probability of having diabetes (0-1)
        """

        if not self._loaded:
            self._load()

        patient_data_scaled = self.scaler.transform(patient_data)

        prediction = self.model.predict(patient_data_scaled)[0]
        probability = self.model.predict_proba(patient_data_scaled)[0][1]

        return prediction, probability