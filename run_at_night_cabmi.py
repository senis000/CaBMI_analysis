__author__ = 'senis'

# this is a program to run at night and go home leaving the computer to work for me...

import pipeline as pipe


def tonight():
    #analyze_tonight()
    cut_tonight()
    separate_me_tonight()
    

def cut_tonight():
    folder = 'G:/Nuria_data/CaBMI/Layer_project/'
    animal = 'IT1'
    days = ['181001', '181002']
    lens = [15000, 38000]
    
    for ind, day in enumerate(days):
        pipe.put_together(folder, animal, day, toplot=False, tocut=True, len_experiment=lens[ind])
    
    
def analyze_tonight():
    
    folder = 'G:/Nuria_data/CaBMI/Layer_project/'
    
#     animal = 'IT3'
#     days = ['181018', '181101']
#     
#     for day in days:
#         print('runing animal: ' + str(animal) + "and day: " + str(day))
#         pipe.put_together(folder, animal, day)
        
        
#     animal = 'IT1'
#     days = ['180928', '181001', '181002', '181018', '181101']
#     
#     for day in days:
#         print('runing animal: ' + str(animal) + " and day: " + str(day))
#         pipe.put_together(folder, animal, day)
#         
#     
#     animal = 'IT2'
#     days = ['181116']
#     
#     for day in days:
#         print('runing animal: ' + str(animal) + " and day: " + str(day))
#         pipe.put_together(folder, animal, day)
        
    animal = 'PT7'
    days = ['181219']
    
    for day in days:
        print('runing animal: ' + str(animal) + " and day: " + str(day))
        pipe.put_together(folder, animal, day)
        
        
def separate_me_tonight():
    
    folder = 'G:/Nuria_data/CaBMI/Layer_project/'
#     animal = 'IT1'
#     
#     days = ['181004', '181005', '181015', '181016', '181017', '181031', '181102', '181113', '181114', '181115', '181116', '190103']
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
#         readme.write("len_base = " + str(len_base)+ '; \n')
#         readme.write("len_bmi = " + str(len_bmi)+ '; \n')
#         readme.close()
        
        
    animal = 'IT2'
    
    days = ['180928', '181001', '181002', '181003', '181015', '181016', '181017', '181018', '181101', '181113', '18115', '190103', '190115', '190117']
    
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
        readme.write("len_base = " + str(len_base)+ '; \n')
        readme.write("len_bmi = " + str(len_bmi)+ '; \n')
        readme.close()
        

        
    animal = 'IT2'
     
    days = ['181004', '181005', '181031', '181102', '190104', '190108', '190109', '190110', '190111', '190116']
     
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
        
    