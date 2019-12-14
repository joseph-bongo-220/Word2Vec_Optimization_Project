import numpy as np 
import re
import nltk
from nltk.corpus import brown
import pickle
import text_funcs as tf
import json

def lambda_handler(event, context):
    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps(tf.get_input_output())
    }
