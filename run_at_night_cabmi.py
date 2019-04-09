__author__ = 'senis'

# this is a program to run at night and go home leaving the computer to work for me...

import pipeline as pipe
import numpy as np
import imp
import shutil
import sys, traceback


def tonight():
    #cut_tonight()
    #analyze_tonight()
    
#     separate_me_tonight()
#     
    #analyze_tonight()
    tonight_caiman()
    

def cut_tonight():
    folder = 'G:/Nuria_data/CaBMI/Layer_project/'
    animal = 'IT2'
    days = ['190109']
    lens = [45000]
    
    for ind, day in enumerate(days):
        pipe.put_together(folder, animal, day, toplot=False, tocut=True, len_experiment=lens[ind])
    
    
def analyze_tonight():
    
    folder = 'G:/Nuria_data/CaBMI/Layer_project/'
    
#     animal = 'IT3'
#     days = ['181004', '181017', '181031']
#      
#     for day in days:
#         print('runing animal: ' + str(animal) + "and day: " + str(day))
#         pipe.put_together(folder, animal, day)
        
        
    animal = 'IT2'
    days = ['180928', '181001', '181002', '181003', '181004', '181005', '181015', '181016', '181017', '181018', '181031', '181101',
            '181102', '181113', '181115', '190103', '190104', '190108', '190109', '190110', '190111', '190115', '190116']
    #
     
    for day in days:
        print('runing animal: ' + str(animal) + " and day: " + str(day))
        pipe.put_together(folder, animal, day)
    
       
def separate_me_tonight():
    
    folder = 'G:/Nuria_data/CaBMI/Layer_project/'
    animal = 'IT1'
     
    days = ['181004', '181005', '181015', '181016', '181017', '181031', '181102', '181113', '181114', '181115', '181116', '190103']
     
    for day in days:
        print('runing animal: ' + str(animal) + " and day: " + str(day))
        fbase = [folder + 'raw/' + animal + '/' + day + '/'+ 'baseline_00001.tif']
        ffull = [folder + 'raw/' + animal + '/' + day + '/'+ 'bmi_00001.tif']
        num_files_b, len_base = pipe.separate_planes(folder, animal, day, fbase, 'baseline')
        num_files, len_bmi = pipe.separate_planes(folder, animal, day, ffull, 'bmi')
         
        nam = folder + 'raw/' + animal + '/' + day + '/' + 'readme.txt'
        readme = open(nam, 'w+')
        readme.write("num_files_b = " + str(num_files_b) + '; \n')
        readme.write("num_files = " + str(num_files)+ '; \n')
        readme.write("len_base = "   + str(len_base)+ '; \n')
        readme.write("len_bmi = " + str(len_bmi)+ '; \n')
        readme.close()
        
#     animal = 'IT6'
#       
#     days = ['190130', '190131', '190212']
#       
#     for day in days:
#         fbase1 = [folder + 'raw/' + animal + '/' + day + '/'+ 'baseline_00001.tif']
#         fbase2 = [folder + 'raw/' + animal + '/' + day + '/'+ 'bmi_00000.tif']
#         ffull = [folder + 'raw/' + animal + '/' + day + '/'+ 'bmi_00001.tif']
#         num_files_b, len_base = pipe.separate_planes_multiple_baseline(folder, animal, day, fbase1, fbase2)
#         num_files, len_bmi = pipe.separate_planes(folder, animal, day, ffull, 'bmi')
#           
#         nam = folder + 'raw/' + animal + '/' + day + '/' + 'readme.txt'
#         readme = open(nam, 'w+')
#         readme.write("num_files_b = " + str(num_files_b) + '; \n')
#         readme.write("num_files = " + str(num_files)+ '; \n')
#         readme.write("len_base = " + str(len_base)+ '; \n')
#         readme.write("len_bmi = " + str(len_bmi)+ '; \n')
#         readme.close()
        

def tonight_caiman():
    folder = 'G:/Nuria_data/CaBMI/Layer_project/'
    animal = 'IT1'
     
    days = ['181004', '181005', '181015', '181016', '181017', '181031', '181102', '181113', '181114', '181115', '181116', '190103']
     
    for day in days:
        folder_path = folder + 'raw/' + animal + '/' + day + '/'
        err_file = open(folder_path + "errlog.txt", 'a+')  # ERROR HANDLING
        vars = imp.load_source('readme', folder_path + 'readme.txt')
        try:
            pipe.analyze_raw_planes(folder, animal, day, vars.num_files, vars.num_files_b, 4, False)
        except Exception as e:
            tb = sys.exc_info()[2]
            err_file.write("\n{}\n".format(folder_path))
            err_file.write("{}\n".format(str(e.args)))
            traceback.print_tb(tb, file=err_file)
            err_file.close()
            sys.exit('Error in analyze raw')
         
        try:
            shutil.rmtree(folder + 'raw/' + animal + '/' + day + '/separated/')
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
 
 
    animal = 'IT2'
    days = ['190117']
     
     
    for day in days:
        folder_path = folder + 'raw/' + animal + '/' + day + '/'
        err_file = open(folder_path + "errlog.txt", 'a+')  # ERROR HANDLING
        vars = imp.load_source('readme', folder_path + 'readme.txt')
        try:
            pipe.analyze_raw_planes(folder, animal, day, vars.num_files, vars.num_files_b, 4, False)
        except Exception as e:
            tb = sys.exc_info()[2]
            err_file.write("\n{}\n".format(folder_path))
            err_file.write("{}\n".format(str(e.args)))
            traceback.print_tb(tb, file=err_file)
            err_file.close()
            sys.exit('Error in analyze raw')
         
        try:
            shutil.rmtree(folder + 'raw/' + animal + '/' + day + '/separated/')
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

            
    animal = 'PT7'
    days = ['181130', '181203', '181204', '181205', '181211', '181212', '181213', '181218', '190114', '190115',
            '190116', '190117', '190118']
     
     
    for day in days:
        folder_path = folder + 'raw/' + animal + '/' + day + '/'
        err_file = open(folder_path + "errlog.txt", 'a+')  # ERROR HANDLING
        vars = imp.load_source('readme', folder_path + 'readme.txt')
        try:
            pipe.analyze_raw_planes(folder, animal, day, vars.num_files, vars.num_files_b, 4, False)
        except Exception as e:
            tb = sys.exc_info()[2]
            err_file.write("\n{}\n".format(folder_path))
            err_file.write("{}\n".format(str(e.args)))
            traceback.print_tb(tb, file=err_file)
            err_file.close()
            sys.exit('Error in analyze raw')
         
        try:
            shutil.rmtree(folder + 'raw/' + animal + '/' + day + '/separated/')
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
            
            