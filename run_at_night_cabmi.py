__author__ = 'senis'

# this is a program to run at night and go home leaving the computer to work for me...

import pipeline_v3 as pipe


def tonight():
    folder = 'C:/Data/Nuria/CaBMI/layer_project/'
    pipe.all_run(folder, 'GCP2', '180725')
    pipe.all_run(folder, 'GCST1', '180727')