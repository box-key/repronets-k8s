import mlflow
from mlflow.pyfunc import PythonModel


class Transliterator(PythonModel):

    def predict(self, context, data_input):
        return "This is a place holder class"


name = 'trf-jpn'
mlflow.set_tracking_uri('postgresql://mlflow-svc:passphrase123@104.248.53.68/mlflow')
if mlflow.get_experiment_by_name(name) is None:
    mlflow.create_experiment(name, artifact_location="sftp://mlflow-svc:09p%59BHNLvRFWtf@104.248.53.68:/uploads/mlflow")
mlflow.set_experiment(name)
mlflow.pyfunc.log_model("py_model", python_model=Transliterator(), artifacts={"models": "/tmp/ctranslate2_released"}) 
