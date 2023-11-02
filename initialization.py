# -*- coding: utf-8 -*-
# code warrior: Barid
##########
import numpy as np
import tensorflow as tf
import configuration
from UNIVERSAL.basic_optimizer import  GradientTowerOpt
from UNIVERSAL.training_and_learning import callback_training
from UNIVERSAL.MLM.preprocess_MLM import preprocess_MLM as preprocess_MLM_fn
from UNIVERSAL.model import dataset_model

import os, sys
from UNIVERSAL.MLM import MASS, BERT, XLM
import Glo_MLM

# import json
# from ast import literal_eval
from numpy import load

cwd = os.getcwd()
co_mat = load("globle_cooccurrence.npy")
print("co-ocurrence matrix loaded")

#############

def glo_fn(labels, input_ids):
    labels = labels.numpy()
    input_ids = input_ids.numpy()
    glo_label = []
    zero_label = np.zeros_like(labels).tolist()
    for i in range(len(labels)):
        if labels[i] == 0:
            glo_label.extend(zero_label)
        else:
            for j in range(len(labels)):
                glo_label.append(co_mat[int(labels[i]), int(input_ids[j])])
    return [glo_label]


def GloMLM_masking(
    inputs,
    configuration
):
    """_summary_

    Args:
        input_ids : integer sequence
        all_special_id : non-masing ids like padding, SOS, EOS
        masking_id : mask id Defaults to 4.
        mlm_probability: how many tokens to be masked Defaults to 0.15.
        mlm_ration: [MASK,  random , unchange] Defaults to [0.8,0.1,0.1].
        label_nonmasking: mark unpredictable token

    Returns:
        masked_inputs, labels
    """
    inputs = inputs.to_tensor()
    # ids = tf.gather(ids, [0], axis=1)
    ids = inputs[0,0:1]
    x = inputs[1,:]
    raw = x
    x_input_span, x_output_span, x_span, x_label = XLM.XLM_masking(
    x,
    configuration.parameters["vocabulary_size"],
    [
        configuration.parameters["PAD_ID"],
        # configuration.parameters["SOS_ID"],
        # configuration.parameters["EOS_ID"],
        configuration.parameters["UNK_ID"],
    ],
    configuration.parameters["MASK_ID"],
    mlm_probability=configuration.parameters['mlm_probability'],
)
    glo_label = tf.py_function(glo_fn, [x_label, raw], tf.float32)
    glo_label.set_shape([None])
    return x_input_span, x_input_span, glo_label, x_label,ids
def Glo_MLM_data_model(parameters):
    return (lambda inputs: GloMLM_masking(inputs, parameters))
