from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model_config import get_config
from neural_net import SkipGram_NN

config = get_config()

app = Flask(__name__)
api = Api(app)

# create new model object
model = SkipGram_NN(layers = [400, 300, 100], step_size = .01, optimizer = "adam", cache = config["cache"])
model.train(batch_size = config["batch_size"])

class Embed(Resource):
    def get(self, text):
        embedding = model.embed(text)
        return {"embedding": list(embedding), "status": 200}

class Train(Resource):
    def get(self):
        return {"status": "TBD"}

# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(Embed, '/embed/<string:text>')
api.add_resource(Train, '/train')

if __name__ == '__main__':
    app.run(debug=False)