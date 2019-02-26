__author__ = 'senis'

# this is a program to run at night and go home leaving the computer to work for me...

import pipeline as pipe


def tonight():
    folder = 'C:/Data/Nuria/CaBMI/layer_project/'
    pipe.all_run(folder, 'GCP2', '180725')
    pipe.all_run(folder, 'GCST1', '180727')
    
    
def analyze_tonight():
#     folder = 'G:/Nuria_data/CaBMI/Layer_project/'
#     animal = 'IT2'
#     day = '181001'
#     
#     pipe.put_together(folder, animal, day)
    
    animal = 'PT7'
    days = ['181126', '181127','181129']
    
    for day in days:
        pipe.put_together(folder, animal, day)
    
    