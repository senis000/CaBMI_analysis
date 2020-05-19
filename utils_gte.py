import re
import os
import sys
import subprocess
import shutil
import numpy as np
import pdb
import matplotlib.pyplot as plt
import pickle
from scipy.stats import zscore
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

def write_params_to_ctrl_file(parameters, control_file_name):
    """
    Writes parameters to a control file.    

    Input:
        PARAMETERS: a dictionary; contains parameters to overwrite the default 
            GTE params
        CONTROL_FILE_NAME: a String; the path to the control.txt file to write
    """
    with open(control_file_name, "w+") as f:
        for key in parameters.keys():
            f.write(key + " = " + str(parameters[key]) + ";\n")

def write_signal_to_file(signal, idx, frame_size, signal_file_name,
        exclude_file_name):
    """
    Writes given neural signals to a signal file.

    Input:
        SIGNAL: a Numpy array of the  neural signal, (num_neurons x num_frames)
        IDX: an integer; the frame index to start writing from
        FRAME_SIZE: an integer; the number of frames of signal to write, 
            starting from IDX
        SIGNAL_FILE_NAME: a String; the path to the signal file to write to 
        EXCLUDE_FILE_NAME: a String; the path to a file to write the indices of
            neurons with a flat signal. These neurons will be excluded. 
    """

    flat_signal_idxs = [] 
    for i in range(signal.shape[0]):
        signal_window = signal[i,idx:idx+frame_size]
        if np.max(signal_window) == np.min(signal_window):
            signal[i,idx] += 0.1
            flat_signal_idxs.append(i)
    with open(signal_file_name, "w+") as f:
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
    with open(exclude_file_name, 'wb') as fp:
        pickle.dump(flat_signal_idxs, fp)

def parse_mathematica_list(file_name):
    """
    Parses a mathematica file into a numpy array.

    Input:
        FILE_NAME: a String; the path to the file containing a mathematica list
    Output:
        CONNECTIVITY_MATRIX: the corresponding Numpy array
    """
    with open(file_name) as f:
        x = f.read()    # Gets the whole mathematica array
    x1 = x[1:-2]    #Strip off the outer brackets

    matches = re.findall('{(.*?)}\\n', x1, flags=0) # Collects each matrix row
    matrix = np.array([eval(m.replace('{', '[').replace('}', ']')) for m in matches], dtype=np.float)
    if matrix.shape[-1] == 1:
        matrix.reshape(matrix.shape[:2])
    connectivity_matrix = matrix
    return connectivity_matrix

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on bottom.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def create_gte_input_files_sliding(
        exp_name, exp_data, parameters,
        frame_size, frame_step=1):
    """
    Given the input, this function will create the necessary directories,
    control files, and signal files that defines an input to the GTE library.
    GTE is assumed to be run in a 'sliding' fashion over the whole signal.

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
        EXCLUDE_FILE_NAMES: an array of Strings. Each String is a path to a
            exclude.p file, itself an array of integer indices.
        OUTPUT_FILE_NAMES: an array of Strings. Each String is a path to a 
            output.mx file, itself a mathematica connectivity matrix.
    """

    try:
        exp_path = "./te-causality/transferentropy-sim/experiments/" + exp_name 
        os.mkdir(exp_path)
        os.mkdir(exp_path + "/outputs")
    except OSError:
        msg = ("Experiment name already exists in GTE experiments folder. " + \
            "Remove the existing directory or rename your experiment to " + \
            "ensure conflicts do not arise.")
        sys.exit(msg)
    control_file_names = []
    exclude_file_names = []
    output_file_names = []
    num_neurons = exp_data.shape[0]
    num_frames = exp_data.shape[1]
    signal = exp_data
    for idx in range(frame_size, num_frames-frame_size, frame_step):
        # Set up the necessary variables and parameters
        control_file_name = exp_path + "/control" + str(idx) + ".txt"
        signal_file_name = exp_path + "/signal" + str(idx) + ".txt"
        exclude_file_name = exp_path + "/exclude" + str(idx) + ".p"
        output_file_name = exp_path + "/outputs/result" + str(idx) + ".mx"
        parameter_file_name = exp_path + "/outputs/parameter" + str(idx) + ".mx" 
        parameters["size"] = num_neurons
        parameters["samples"] = frame_size
        parameters["inputfile"] = "\"" + signal_file_name + "\""
        parameters["outputfile"] = "\"" + output_file_name + "\""
        parameters["outputparsfile"] = "\"" + parameter_file_name + "\"" 
        # Generate the CONTROL.TXT and SIGNAL.TXT file. Save the file path of 
        # the control file and the result file (which is not yet generated).
        write_params_to_ctrl_file(parameters, control_file_name)
        write_signal_to_file(signal, idx, frame_size,
            signal_file_name, exclude_file_name)
        control_file_names.append(control_file_name)
        exclude_file_names.append(exclude_file_name)
        output_file_names.append(output_file_name)
    return control_file_names, exclude_file_names, output_file_names


def create_gte_input_files(exp_name, exp_data,
        parameters, to_zscore=False, zscore_threshold=0.0):
    """
    Given the input, this function will create the necessary directories,
    control files, and signal files that defines an input to the GTE library.
    GTE is run over each matrix in EXP_DATA

    Input:
        EXP_NAME: a String; the path to the HDF5 file of the experiment
        EXP_DATA: a 3D numpy array of size (trials x neurons x frames)
        PARAMETERS: a dictionary; contains parameters to overwrite the default
            GTE params. Users do not need to define 'size', 'samples',
            'inputfile', 'outputfile', 'outputparsfile'-- defined values 
            for these parameters will be overwritten
        TO_ZSCORE: a boolean flag indicating whether the data needs to be
            zscored and thresholded (at ZSCORE_THRESHOLD).
    Output:
        CONTROL_FILE_NAMES: an array of Strings. Each String is a path to a 
            control.txt file, itself an input to the GTE library.
        EXCLUDE_FILE_NAMES: an array of Strings. Each String is a path to a
            exclude.p file, itself an array of integer indices.
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
        sys.exit(msg)

    control_file_names = []
    exclude_file_names = []
    output_file_names = []
    num_trials = exp_data.shape[0]
    num_neurons = exp_data.shape[1]
    num_frames = exp_data.shape[2]
    for idx in range(num_trials):
        # Set up the necessary variables and parameters
        signal = exp_data[idx,:,:]
        signal_start = np.argwhere(~np.isnan(signal[0,:]))[0,0]
        signal = signal[:,signal_start:]

        # Check the trial is long enough to run GTE. Arbitrarily,
        # it should be at least 30 frames long.
        if signal.shape[1] < 30:
            continue

        # Process the signal
        if to_zscore:
            signal = zscore(signal, axis=1)
            signal = np.nan_to_num(signal)
            signal = np.maximum(signal, -1.0*zscore_threshold)
            signal = np.minimum(signal, zscore_threshold)
        control_file_name = exp_path + "/control" + str(idx) + ".txt"
        signal_file_name = exp_path + "/signal" + str(idx) + ".txt"
        exclude_file_name = exp_path + "/exclude" + str(idx) + ".txt"
        output_file_name = exp_path + "/outputs/result" + str(idx) + ".mx"  
        parameter_file_name = exp_path + "/outputs/parameter" + str(idx) + ".mx" 
        parameters["size"] = num_neurons
        parameters["samples"] = signal.shape[1]
        parameters["inputfile"] = "\"" + signal_file_name + "\""
        parameters["outputfile"] = "\"" + output_file_name + "\""
        parameters["outputparsfile"] = "\"" + parameter_file_name + "\"" 
        # Generate the CONTROL.TXT and SIGNAL.TXT file. Save the file path of 
        # the control file and the result file (which is not yet generated).
        write_params_to_ctrl_file(parameters, control_file_name)
        write_signal_to_file(
            signal, 0, signal.shape[1], signal_file_name, exclude_file_name
            )
        control_file_names.append(control_file_name)
        exclude_file_names.append(exclude_file_name)
        output_file_names.append(output_file_name)
    return control_file_names, exclude_file_names, output_file_names


def run_gte(control_file_names, exclude_file_names, output_file_names, method='te-extended'):
    """
    Runs GTE on each control file.

    Input:
        CONTROL_FILE_NAMES: an array of Strings. Each String is a path to a 
            control.txt file that should be run. 
        EXCLUDE_FILE_NAMES: an array of Strings. Each String is a path to a
            exclude.p file, itself an array of integer indices.
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
        print(control_file_name)
        exe_code = subprocess.call([
            f"./te-causality/transferentropy-sim/{method}", control_file_name
            ])
        result = parse_mathematica_list(output_file_names[idx])
        exclude_file_name = exclude_file_names[idx]
        with open(exclude_file_name, 'rb') as p_file:
            exclude_idxs = pickle.load(p_file)
        for idx in exclude_idxs:
            result[idx,:] = np.nan
            result[:,idx] = np.nan
        results.append(result)
    return results

def visualize_gte_results(results, neuron_locations, cmap='r'):
    """
    This function will make an animated 3d scatterplot of the neurons. It will
    then show neural connectivity changing over time, as defined by RESULTS.
    Input:
        RESULTS: an array of connectivity matrices in temporal sequence.
            The matrix is square, where the i,jth entry corresponds to the 
            transfer of information from neuron i to neuron j. 
        NEURON_LOCATIONS: a Numpy array of the spatial locations of each neuron;
            using the CaBMI notation, this is the COM_CM variable. This array is
            (num_neurons x num_dim) in size. The dimensions are in (x,y,z) order
        CMAP: Color or sequence of colors to pass into the scatter
    """ #TODO: Add save_video flag via anim.save("anim.mp4", fps=1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        neuron_locations[:,0], neuron_locations[:,1], neuron_locations[:,2],
        c=cmap
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
        ax.text2D(0.05, 0.95, "Frame " + str(i),
            transform=ax.transAxes, color="blue"
            )
        return ax

    anim = animation.FuncAnimation(fig, animate, frames=range(0,len(results)),
                                   init_func=init, interval=500, blit=False)
    plt.show()

def visualize_gte_matrices(results, labels=None, cmap="YlGn"):
    """
    This function will make heatmaps of the connectivity matrices, adjustable
    by a slider bar.
    Input:
        RESULTS: an array of connectivity matrices in temporal sequence.
            The matrix is square, where the i,jth entry corresponds to the 
            transfer of information from group i to group j. 
        LABELS: a 1D Numpy array; contains the labels for each group in results.
            If not provided, the default is to label each group numerically,
            by 0-indexing.
        CMAP: a String; the colormap to use for IMSHOW (the matrix heat map) 
    """ #TODO: Add save_video flag via anim.save("anim.mp4", fps=1)
    max_val = max([np.max(np.nan_to_num(m)) for m in results])
    min_val = min([np.min(np.nan_to_num(m)) for m in results])
    num_neurons = results[0].shape[0]
    if labels is None:
        labels = np.arange(num_neurons)
    
    # Initial plot
    fig, ax = plt.subplots(nrows=1, ncols=1)
    im = ax.imshow(results[0], cmap=cmap, vmin=min_val, vmax=max_val)
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.subplots_adjust(bottom=0.25, top=0.825)
    ax.set_title('Change in Transfer Entropy Over Time')

    axcolor = 'lightgoldenrodyellow'
    axtrials = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor=axcolor)
    trial_slider = Slider(axtrials, 'Frame', 0, len(results)-1, valinit=0)
    def update(val, max_val=max_val, min_val=min_val):
        trial = int(trial_slider.val)
        ax.clear()
        im = ax.imshow(results[trial], cmap=cmap, vmin=min_val, vmax=max_val)
        ax.set_title('Change in Transfer Entropy Over Time')
        fig.canvas.draw_idle()
    trial_slider.on_changed(update)
    plt.show(block=True)

def compare_gte_matrices(result1, result2, labels=None, cmap="YlGn"):
    """
    This function will make heatmaps of to compare two GTE matrices,
    side-by-side. Assumes both matrices are the same size.
    Input:
        RESULT1: A square numpy matrix where the i,jth entry corresponds to the 
            transfer of information from group i to group j.
        RESULT2: A square numpy matrix where the i,jth entry corresponds to the 
            transfer of information from group i to group j. 
        LABELS: a 1D Numpy array; contains the labels for each group in results.
            If not provided, the default is to label each group numerically,
            by 0-indexing.
        CMAP: a String; the colormap to use for IMSHOW (the matrix heat map) 
    """ #TODO: Add save_video flag via anim.save("anim.mp4", fps=1)
    max_val = max([np.max(np.nan_to_num(m)) for m in [result1, result2]])
    min_val = min([np.min(np.nan_to_num(m)) for m in [result1, result2]])
    num_neurons = result1.shape[0]
    if labels is None:
        labels = np.arange(num_neurons)
    
    # Make basic imshow plot
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(7,5))
    im1 = ax1.imshow(result1, cmap=cmap, vmin=min_val, vmax=max_val)
    im2 = ax2.imshow(result2, cmap=cmap, vmin=min_val, vmax=max_val)
    cbar = ax2.figure.colorbar(im2, ax=ax2)
    # Add in labels
    for ax in [ax1, ax2]:
        ax.set_xticks(np.arange(labels.size))
        ax.set_yticks(np.arange(labels.size))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    return fig, (ax1, ax2)

def delete_gte_files(dir_names=None, remove_only_input_files=False):
    """
    Deletes GTE directories in files created to run the GTE library.
    Input:
        DIR_NAMES: A list; the names of specific directories to remove from
            /te-causality/transferentropy-sim/experiments/. If not provided,
            the function by default removes all directories.
        REMOVE_ONLY_INPUT_FILES: If True, only the parameter, control, and
            exclude files are removed from the directories in DIR_NAMES.
    """

    exp_dir = "./te-causality/transferentropy-sim/experiments"
    if dir_names is None:
        dir_names = os.listdir(exp_dir) # Assumes only directories exist here
    for dir_name in dir_names: # Loop through each directory found
        dir_path = exp_dir + "/" + dir_name
        if remove_only_input_files:
            for f in os.listdir(dir_path):
                if f.endswith(".txt") or f.endswith(".p"):
                    subprocess.call(["rm", "-rf", dir_path + "/" + f])
        else:
            shutil.rmtree(dir_path)


def group_result(result, grouping, ignore_diagonal=True):
    '''
    Groups an existing GTE connectivity matrix by averaging scores
    over user-defined groupings of neurons.
    Inputs:
        RESULT: A numpy matrix (GTE connectivity matrix)
        GROUPING: Numpy array of NUM_NEURONS size; an integer ID is given
            to each neuron, defining their group. This array is 0-indexed.
    Outputs:
        GROUPED_RESULT: If GROUPING has N unique values, GROUPED_RESULT is
            a numpy matrix of dimension (N, N)
    '''
    num_neurons = result.shape[0]
    num_groups = np.unique(grouping).size
    if grouping.size != num_neurons:
        raise RuntimeError('Wrong dimensions for GROUPING')

    grouped_result = np.zeros((num_groups, num_groups))
    for i in range(num_groups):
        for j in range(num_groups):
            if (i == j) and ignore_diagonal:
                continue
            relevant_vals = []
            for k in range(num_neurons): # Find k in group i
                if grouping[k] != i:
                    continue
                for l in range(num_neurons): # Compare to l in group j
                    if (grouping[l] != j) or (k==l):
                        continue
                    relevant_vals.append(result[k, l])
            grouped_result[i, j] = np.nanmean(relevant_vals)
    return grouped_result
