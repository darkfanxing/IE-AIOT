from keras.models import model_from_json
from utility import get_data, change_to_sequence_data
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="darkgrid")

def load_model(model_name, model_weights_filename):
  with open("model/{}.json".format(model_name), "r") as json_file:
    model = model_from_json(json_file.read())

  model.load_weights("model/{}.h5".format(model_weights_filename))
  model.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["categorical_accuracy"])
              
  return model

if __name__ == "__main__":
  model = load_model("new-CNN-RNN", "new")
  
  # Change your file path
  data_path = "./data/unfocused/kuo_1.csv"
  data = get_data(data_path)
  data = change_to_sequence_data(data)

  predictions = model.predict(data)
  # predictions = np.where(predictions==1)[1]
  
  sns.lineplot(x=[time for time in range(predictions.shape[0])], y=predictions[:, 0]).set(xlabel="time(s)", ylabel="focused_score")