from src.pipelines.prediction_pipeline import PredictionPipeline
from pywebio.input import *
from pywebio.output import *
import numpy as np

# building the interface
data = input_group("Predict the ", [
    select(
        label = "Select the type of job",
        options = [
            "services", "blue-collar", "technician", "management", "student", "self-employed",\
                "housemaid", "retired", "entrepreneur", "admin.", "unemployed"
        ],
        required = True,
        name = "job"
    ),
    select(
        label = "Select month",
        options = [
            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
        ],
        required = True,
        name = "month"
    ),
    input("Enter average yearly balance (in euros)", name = "balance", required = True, type = FLOAT),
    input("Enter last contact duration (in seconds)", name = "duration", required = True, type = NUMBER),
    input("Enter number of contacts performed for this customer in current campaign", name = "campaign",\
          required = True, type = NUMBER),
    radio("Select outcome of the previous campaign for this customer", options = ["success", "failure", "not applicable"], required = True,\
          name = "poutcome", inline = True),
    radio("Was the customer ever contacted previously", options = ["yes", "no"], inline = True,
          required = True, name = "contacted_previously"),
    select("Type of loan(s) the customer has/have", options = ["personal loan only", "housing loan only", "both"],\
           required = True, name = "loans")
])

# encoding few features manually
### poutcome
if data["poutcome"] == "success":
    data["poutcome"] = 1
else: data["poutcome"] = 0

### contacted_previously
if data["contacted_previously"] == "yes":
    data["contacted_previously"] = 1
else: data["contacted_previously"] = 0

### loans
if data["loans"] == "personal loan only":
    data["loans"] = 1
elif data["loans"] == "housing loan only":
    data["loans"] = 1
elif data["loans"] == "both":
    data["loans"] = 2
else: data["loans"] = 0

if __name__ == "__main__":
    predictionPipeline = PredictionPipeline()
    predictionPipeline.predictResult(array = np.array(data.values()))