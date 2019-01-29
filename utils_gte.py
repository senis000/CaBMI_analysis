import re
import os
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

def write_params_to_ctrl_file(parameters, control_file_name):
    """
    Writes parameters to a control file.    

    Input:
        PARAMETERS: a dictionary; contains parameters to overwrite the default 
            GTE params
        CONTROL_FILE_NAME: a String; the path to the control.txt file to write
    """
    f = open(control_file_name, "w+") 
    for key in parameters.keys():
        f.write(key + " = " + str(parameters[key]) + ";\n")

def write_signal_to_file(signal, idx, frame_size, signal_file_name):
    """
    Writes given neural signals to a signal file.

    Input:
        SIGNAL: a Numpy array of the  neural signal, (num_neurons x num_frames)
        IDX: an integer; the frame index to start writing from
        FRAME_SIZE: an integer; the number of frames of signal to write, 
            starting from IDX
        SIGNAL_FILE_NAME: a String; the path to the signal file to write to 
    """

    f = open(signal_file_name, "w+")
    num_neurons = signal.shape[0]
    num_frames = frame_size
    for i in range(idx, idx+frame_size):
        line = ""
        for j in range(num_neurons):
            if j==0:
                line += str(signal[j,i])
            else:
                line+=(","+str(signal[j,i]))
        f.write(line+"\n") 

def parse_mathematica_list(file_name):
    """
    Parses a mathematica file into a numpy array.

    Input:
        FILE_NAME: a String; the path to the file containing a mathematica list
    Output:
        CONNECTIVITY_MATRIX: the corresponding Numpy array
    """

    f = file(file_name)
    x = f.read()    # Gets the whole mathematica array
    x1 = x[1:-2]    #Strip off the outer brackets
    matches = re.findall('{(.*?)}\\n', x1, flags=0) # Collects each matrix row
    matrix = [m.split(', ') for m in matches]
    matrix = [np.array(row).astype(np.float) for row in matrix]
    connectivity_matrix = np.array(matrix)
    return connectivity_matrix

def create_gte_input_files(
        exp_name, exp_data, parameters,
        frame_size, frame_step=1):
    """
    Given the input, this function will create the necessary directories,
    control files, and signal files that defines an input to the GTE library.

    Input:
        EXP_NAME: a String; the path to the HDF5 file of the experiment
        EXP_DATA: a numpy array; a (num_neurons x num_frames) matrix
        PARAMETERS: a dictionary; contains parameters to overwrite the default
            GTE params. Users do not need to define 'size', 'samples',
            'inputfile', 'outputfile', 'outputparsfile'-- defined values 
            for these parameters will be overwritten
        FRAME_SIZE: an integer; the number of frames we process with GTE. 
            The frame size will be 'slid' over the whole signal.
        FRAME_STEP: an integer; the size of the step to take through the signal.
            For instance, frame_step=1 resembles a convolution.
    Output:
        CONTROL_FILE_NAMES: an array of Strings. Each String is a path to a 
            control.txt file, itself an input to the GTE library.
        OUTPUT_FILE_NAMES: an array of Strings. Each String is a path to a 
            output.mx file, itself a mathematica connectivity matrix.
    """

    try:
        exp_path = "./te-causality/transferentropy-sim/experiments/" + exp_name 
        os.mkdir(exp_path)
        os.mkdir(exp_path + "/outputs")
    except OSError:
        msg = ("Experiment name already exists in GTE experiments folder. "
               "Remove the existing directory or rename your experiment to "
               "ensure conflicts do not arise.")
	#sys.exit(msg)
    control_file_names = []
    output_file_names = []
    num_neurons = exp_data.shape[0]
    num_frames = exp_data.shape[1]
    signal = exp_data
    for idx in range(frame_size, num_frames-frame_size, frame_step):
        # Set up the necessary variables and parameters
        pdb.set_trace()
        control_file_name = exp_path + "/control" + str(idx) + ".txt"
        signal_file_name = exp_path + "/signal" + str(idx) + ".txt"
        output_file_name = exp_path + "/outputs/result" + str(idx) + ".mx"  
        parameter_file_name = exp_path + "/outputs/parameter" + str(idx) + ".mx" 
        parameters["size"] = num_neurons
        parameters["samples"] = frame_size
        parameters["inputfile"] = "\"" + signal_file_name + "\""
        parameters["outputfile"] = "\"" + output_file_name + "\""
        parameters["outputparsfile"] = "\"" + parameter_file_name + "\"" 
        # Generate the CONTROL.TXT and SIGNAL.TXT file. Save the file path of 
        # the control file and the result file (which is not yet generated).
        #write_params_to_ctrl_file(parameters, control_file_name)
        #write_signal_to_file(signal, idx, frame_size, signal_file_name)
        control_file_names.append(control_file_name)
        output_file_names.append(output_file_name)
    return control_file_names, output_file_names

def run_gte(control_file_names, output_file_names, pickle_results):
    """
    Runs GTE on each control file.

    Input:
        CONTROL_FILE_NAMES: an array of Strings. Each String is a path to a 
            control.txt file that should be run. 
        OUTPUT_FILE_NAMES: an array of Strings. Each String is a path to a 
            result.mx file that contains the result of running GTE.
    Output:
        RESULTS: an array of connectivity matrices. The ith matrix corresponds 
            to the output of GTE given the ith control file from the input.
            The matrix is square, where the i,jth entry corresponds to the 
            transfer of information from neuron i to neuron j.
    """

    results = []
    for idx, control_file_name in enumerate(control_file_names): # TODO: parallelize
        exe_code = subprocess.call([
            "./te-causality/transferentropy-sim/te-extended", control_file_name
            ])
        result = parse_mathematica_list(output_file_names[idx])
        results.append(result)
    if pickle_results:
        results_path = os.path.dirname(control_file_name) + "/outputs/results.p"
        pickle.dump(results, open(results_path, "wb"))
    return results

def visualize_gte_results(results, neuron_locations, color_map='r'):
    """
    This function will make a 3d scatterplot of the neurons. It will then show
    neural connectivity changing over time, as defined by the RESULTS matrix.
    Input:
        RESULTS: an array of connectivity matrices in temporal sequence.
            The matrix is square, where the i,jth entry corresponds to the 
            transfer of information from neuron i to neuron j. 
        NEURON_LOCATIONS: a Numpy array of the spatial locations of each neuron;
            using the CaBMI notation, this is the COM_CM variable. This array is
            (num_neurons x num_dim) in size. The dimensions are in (x,y,z) order
        COLOR_MAP: Color or sequence of colors to pass into the scatter
            function. Useful for labelling specific neurons. Default
            coloring is red.
    """ #TODO: Add save_video flag via anim.save("anim.mp4", fps=1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        neuron_locations[:,0], neuron_locations[:,1], neuron_locations[:,2],
        c=color_map
        )
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate") 
    ax.set_zlabel("Z Coordinate") 
    
    def init(): # Initializes the background of each frame
        return ax,

    def animate(i): # Animates sequentially by index i
        m = results[i]
        dims = m.shape[0]
        sorted_idxs = np.argsort(m.flatten())
        sorted_idxs_xy = [(idx//dims, idx%dims) for idx in sorted_idxs]
        for l in ax.get_lines():
            ax.lines.remove(l)
        for t in ax.texts:
            t.remove()
            
        num_links = int(dims*dims*.0005)
        for j in range(num_links):
            source = sorted_idxs_xy[-j-1][0]
            sink = sorted_idxs_xy[-j-1][1]
            xs = [neuron_locations[source,0], neuron_locations[sink,0]]
            ys = [neuron_locations[source,1], neuron_locations[sink,1]]
            zs = [neuron_locations[source,2], neuron_locations[sink,2]]
            ax.plot(xs, ys, zs)
        ax.text2D(0.05, 0.95, "Frame " + str(i), transform=ax.transAxes, color="blue")
        return ax

    anim = animation.FuncAnimation(fig, animate, frames=range(0,len(results)),
                                   init_func=init, interval=500, blit=False)
    plt.show()

def delete_gte_files(exp_name, delete_output=True):
    """
    Deletes GTE text files created to run the GTE library.
    Input:
        EXP_NAME: Deletes /te-causality/transferentropy-sim/experiments/EXP_NAME.
            The directory contents are deleted as well.
        DELETE_OUTPUT: A boolean flag, default True. If set to False, the
            function will delete everything inside
            /te-causality/transferentropy-sim/experiments/EXP_NAME, 
            except for the 'output' directory. 
    """

    exp_dir = "./te-causality/transferentropy-sim/experiments/" + exp_name 
    if delete_output:
        shutil.rmtree(exp_dir) 
    else:
        subprocess.call(["rm", "-rf", exp_dir + "/*.txt"])
