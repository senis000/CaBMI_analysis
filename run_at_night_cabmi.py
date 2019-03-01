__author__ = 'senis'

# this is a program to run at night and go home leaving the computer to work for me...

import pipeline as pipe


def tonight():
    folder = 'C:/Data/Nuria/CaBMI/layer_project/'
    pipe.all_run(folder, 'GCP2', '180725')
    pipe.all_run(folder, 'GCST1', '180727')
    
    
def analyze_tonight():
    
    folder = 'G:/Nuria_data/CaBMI/Layer_project/'
    
    animal = 'PT7'
    days = ['181126', '181127','181129']
    
    for day in days:
        pipe.put_together(folder, animal, day)
        
        
def separate_me_tonight():
    
    folder = 'G:/Nuria_data/CaBMI/Layer_project/'
    animal = 'IT2'
    
    days = ['181018', '181101', '181116']
    
    for day in days:
        fbase1 = [folder + 'raw/' + animal + '/' + day + '/'+ 'baseline_00001.tif']
        fbase2 = [folder + 'raw/' + animal + '/' + day + '/'+ 'bmi_00000.tif']
        ffull = [folder + 'raw/' + animal + '/' + day + '/'+ 'bmi_00001.tif']
        num_files_b, len_base = pipe.separate_planes_multiple_baseline(folder, animal, day, fbase1, fbase2)
        num_files, len_bmi = pipe.separate_planes(folder, animal, day, ffull, 'bmi')
        
        nam = folder + 'raw/' + animal + '/' + day + '/' + 'readme.txt'
        readme = open(nam, 'w+')
        readme.write("num_files_b = " + str(num_files_b) + '; \n')
        readme.write("num_files = " + str(num_files)+ '; \n')
        readme.write("len_base = " + str(len_base)+ '; \n')
        readme.write("len_bmi = " + str(len_bmi)+ '; \n')
        readme.close()
        
        
    animal = 'IT1'
    
    days = ['181018', '181101']
    
    for day in days:
        fbase1 = [folder + 'raw/' + animal + '/' + day + '/'+ 'baseline_00001.tif']
        fbase2 = [folder + 'raw/' + animal + '/' + day + '/'+ 'bmi_00000.tif']
        ffull = [folder + 'raw/' + animal + '/' + day + '/'+ 'bmi_00001.tif']
        num_files_b, len_base = pipe.separate_planes_multiple_baseline(folder, animal, day, fbase1, fbase2)
        num_files, len_bmi = pipe.separate_planes(folder, animal, day, ffull, 'bmi')
        
        nam = folder + 'raw/' + animal + '/' + day + '/' + 'readme.txt'
        readme = open(nam, 'w+')
        readme.write("num_files_b = " + str(num_files_b) + '; \n')
        readme.write("num_files = " + str(num_files)+ '; \n')
        readme.write("len_base = " + str(len_base)+ '; \n')
        readme.write("len_bmi = " + str(len_bmi)+ '; \n')
        readme.close()
        
        
    animal = 'PT7'
    
    days = ['181219']
    
    for day in days:
        fbase1 = [folder + 'raw/' + animal + '/' + day + '/'+ 'baseline_00001.tif']
        fbase2 = [folder + 'raw/' + animal + '/' + day + '/'+ 'bmi_00000.tif']
        ffull = [folder + 'raw/' + animal + '/' + day + '/'+ 'bmi_00001.tif']
        num_files_b, len_base = pipe.separate_planes_multiple_baseline(folder, animal, day, fbase1, fbase2)
        num_files, len_bmi = pipe.separate_planes(folder, animal, day, ffull, 'bmi')
        
        nam = folder + 'raw/' + animal + '/' + day + '/' + 'readme.txt'
        readme = open(nam, 'w+')
        readme.write("num_files_b = " + str(num_files_b) + '; \n')
        readme.write("num_files = " + str(num_files)+ '; \n')
        readme.write("len_base = " + str(len_base)+ '; \n')
        readme.write("len_bmi = " + str(len_bmi)+ '; \n')
        readme.close()
        
    