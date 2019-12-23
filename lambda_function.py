import numpy as np 
import re
import nltk
from nltk.corpus import brown
import pickle
import text_funcs as tf
import json

def lambda_handler(event, context):
    # TODO implement
    X, Y, lookup = tf.get_input_output()
    payload = {"input": X.tolist(), "output": Y.tolist(), "lookup": lookup}

    return {
        'statusCode': 200,
        'body': json.dumps(tf.get_input_output())
    }
