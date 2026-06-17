from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class ModelFactory:

    @staticmethod
    def create_models():

        models = {

            "RandomForest":
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                ),

            "SVM":
                SVC(
                    kernel="rbf",
                    probability=True
                ),

            "LogisticRegression":
                LogisticRegression(
                    max_iter=500
                ),

            "MLP":
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    max_iter=1000,
                    random_state=42
                )
        }

        return models