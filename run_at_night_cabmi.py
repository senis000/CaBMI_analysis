__author__ = 'senis'

# this is a program to run at night and go home leaving the computer to work for me...

import pipeline as pipe
import numpy as np
import imp
import shutil, os
import sys, traceback


def tonight():
    put_together_tonight(folder = 'I:/Nuria_data/CaBMI/Layer_project/', animals = ('IT1','IT2', 'IT8','PT7'))


#     folder = 'G:/Nuria_data/CaBMI/Layer_project/'
#     animal = 'PT18'
#     days = ['190729']
#         
#     for day in days:
#         print('runing animal: ' + str(animal) + "and day: " + str(day))
#         pipe.put_together(folder, animal, day)  
#         
#     cut_tonight()
#     analyze_tonight()
#     put_together_tonight()
#     put_together_tonight(folder = 'G:/Nuria_data/CaBMI/Layer_project/', animals =  ['PT13'])
#     separate_me_tonight()
#     
    #analyze_tonight()
    #tonight_caiman()
    

def cut_tonight():
    folder = 'G:/Nuria_data/CaBMI/Layer_project/'
    animal = 'PT18'
    days = ['190723', '190801']
    lens = [34000, 34000]
     
    for ind, day in enumerate(days):
        pipe.put_together(folder, animal, day, toplot=False, tocut=True, len_experiment=lens[ind])
    

    folder = 'J:/Nuria_data/CaBMI/Layer_project/'
    

    animal = 'PT19'
    days = ['190717', '190718', '190720']
    lens = [21000, 21000, 20000]
     
    for ind, day in enumerate(days):
        pipe.put_together(folder, animal, day, toplot=False, tocut=True, len_experiment=lens[ind])
        
    animal = 'IT8'
    days = ['190207', '190205']
    lens = [34000, 31000]
     
    for ind, day in enumerate(days):
        pipe.put_together(folder, animal, day, toplot=False, tocut=True, len_experiment=lens[ind])


    animal = 'IT9'
    days = ['190301']
    lens = [36000]
     
    for ind, day in enumerate(days):
        pipe.put_together(folder, animal, day, toplot=False, tocut=True, len_experiment=lens[ind])
        
        
    animal = 'IT10'
    days = ['190314']
    lens = [24000]
     
    for ind, day in enumerate(days):
        pipe.put_together(folder, animal, day, toplot=False, tocut=True, len_experiment=lens[ind])
    
#     animal = 'PT13'
#     days = ['190121', '190123']
#     lens = [24000, 18000]
#     
#     for ind, day in enumerate(days):
#         pipe.put_together(folder, animal, day, toplot=False, tocut=True, len_experiment=lens[ind])

def put_together_tonight(folder = 'J:/Nuria_data/CaBMI/Layer_project/', animals = ('IT8', 'IT9', 'IT10')):
    
#     
#     animal = 'IT4'
#     days = ['181001']
#       
#     for day in days:
#         print('runing animal: ' + str(animal) + "and day: " + str(day))
#         pipe.put_together(folder, animal, day)       
    
    raw_folder = os.path.join(folder, 'raw')
    processed = os.path.join(folder, 'processed')
    ns = "full_{}_{}__data.hdf5"
    fails = {}
    for animal in animals:
        fails[animal] = []
        animal_path = os.path.join(raw_folder, animal)
        days = []
        
        for d in os.listdir(animal_path):
            if d.isalpha():
                print(d)
                continue
            if os.path.exists(os.path.join(processed, animal, ns.format(animal, d))):
                continue
            else:
                days.append(d)
        print(animal, days)
          
        for day in days:
            target = os.path.join(animal_path, day)
            if not os.path.exists(target):
                print("No target", target)
                continue
            print('runing animal: ' + str(animal) + "and day: " + str(day))
            try:
                pipe.put_together(folder, animal, day)
            except Exception as e:
                fails[animal].append((day, e.args))
    print(fails)       
            
    
def analyze_tonight():
    
    folder = 'J:/Nuria_data/CaBMI/Layer_project/'
#     
#     animal = 'IT4'
#     days = ['181001']
#       
#     for day in days:
#         print('runing animal: ' + str(animal) + "and day: " + str(day))
#         pipe.put_together(folder, animal, day)
    
        

    animal = 'IT8'
     
    days = ['190204', '190205', '190206', '190208', '190214', '190215', '190220', '190221', 
            '190222', '190225', '190227', '190301']
       
    for day in days:
        print('runing animal: ' + str(animal) + "and day: " + str(day))
        pipe.put_together(folder, animal, day)
        
    
#     raw_folder = os.path.join(folder, 'raw')
#     processed = os.path.join(folder, 'processed')
#     ns = "full_{}_{}__data.hdf5"
#    
#     animal = 'IT1'
#     days = ['181102', '181113', '181114', '181115', '181116', '190103']
#     
#     for day in days:
#         target = os.path.join(folder, 'raw', animal, day)
#         if not os.path.exists(target):
#             print("No target", target)
#             continue
#         print('runing animal: ' + str(animal) + "and day: " + str(day))
#         pipe.put_together(folder, animal, day)
        
    
#     animal = 'PT13'
#     
#     days = ['190121', '190123']
#       
#     for day in days:
#         print('runing animal: ' + str(animal) + "and day: " + str(day))
#         pipe.put_together(folder, animal, day)
        
#         
#     animal = 'IT2'
#     days = ['180928', '181001', '181002', '181003', '181004', '181005', '181015', '181016', '181017', '181018', '181031', '181101',
#             '181102', '181113', '181115', '190103', '190104', '190108', '190109', '190110', '190111', '190115', '190116']
#     #
#      
#     for day in days:
#         print('runing animal: ' + str(animal) + " and day: " + str(day))
#         pipe.put_together(folder, animal, day)
    
       
def separate_me_tonight():
    
    folder = 'J:/Nuria_data/CaBMI/Layer_project/'
#     animal = 'IT8'
#        
#     days = ['190121']
#        
#     for day in days:
#         print('runing animal: ' + str(animal) + " and day: " + str(day))
#         fbase = [folder + 'raw/' + animal + '/' + day + '/'+ 'baseline_00001.tif']
#         ffull = [folder + 'raw/' + animal + '/' + day + '/'+ 'bmi_00001.tif']
#         num_files_b, len_base = pipe.separate_planes(folder, animal, day, fbase, 'baseline')
#         num_files, len_bmi = pipe.separate_planes(folder, animal, day, ffull, 'bmi')
#            
#         nam = folder + 'raw/' + animal + '/' + day + '/' + 'readme.txt'
#         readme = open(nam, 'w+')
#         readme.write("num_files_b = " + str(num_files_b) + '; \n')
#         readme.write("num_files = " + str(num_files)+ '; \n')
#         readme.write("len_base = "   + str(len_base)+ '; \n')
#         readme.write("len_bmi = " + str(len_bmi)+ '; \n')
#         readme.close()
#         
    animal = 'IT8'
       
    days = ['190207', '190213']
       
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
        
        
    animal = 'IT9'
       
    days = ['190304', '190308', '190311']
       
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
        
        
    animal = 'IT10'
       
    days = ['190220', '190308']
       
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
            
            