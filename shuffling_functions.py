import numpy as np


def shuffle_peaks(data_array, sf, ef, ipi_proc, axis=0):
    """
    Takes in data_array, any shape, of calcium peaks and shuffle with
    respect to axis; Do this using numpy parallel features.

    Input:
        data_array: ndarray
            Numpy array with peak data and inter-peak intervals
        sf: function
            start function that takes in (data_array, i) where i is the
            index that signifies the start of a peak region (inclusive)
        ef: function
            end function that takes in (data_array, j) where j the index
            that signifies the end of a peak region (exclusive),
            therefore, data_array[i:j] would represent one peak region
        ipi_proc: function
            randomizing procedure for ipi data shuffling
        axis: int
            axis for shuffling

    Returns:
        shuffled: ndarray
            shuffled version of peak array with respect to axis
    """
    # Critical problem could be fitting of the background activity curve
    peak_regions = []
    prev_end = 0
    IPI = np.array(np.empty(data_array.shape[:axis]+(0,)+data_array.shape[
        axis+1:]))
    pk_start = None
    # TODO: PROCEDURE ONLY APPLIES TO 1D NOW
    for i in range(len(data_array)):
        if pk_start is None:
            # Outside of peak regions, wait for criterion sf to be met
            if sf(data_array, i):
                pk_start = i
                IPI = np.concatenate((IPI,
                    np.take(data_array, range(prev_end, pk_start),
                            axis=axis)), axis=axis)
        else:
            # Within peak_regions, record the peak if ef criterion is met
            if ef(data_array, i):
                peak_regions.append((pk_start, i))
                prev_end = i
                pk_start = None

    IPI = np.concatenate((IPI,
        np.take(data_array, range(prev_end, data_array.shape[axis]),
                axis=axis)), axis=axis)

    # TODO: USE NUMPY Parallelization later to expedite the process, so far
    #  use naive method
    np.random.shuffle(peak_regions)
    newIPI = ipi_proc(IPI)
    s_inds = np.random.choice(len(IPI)+1, len(peak_regions))
    # TODO: RETURN SEQUENCE SUCH THAT all peak region sequences will be
    #  appended in accordance


def shuffle_peaks_1d(data_array, ipi_proc, perc=50, debug=False):
    data_array = data_array[~np.isnan(data_array)]
    if debug:
        peak_regions, IPIs = signal_partition(data_array, perc, debug)
    else:
        peak_regions, IPIs, points = signal_partition(data_array, perc, debug)
    IPI = np.concatenate(IPIs)
    np.random.shuffle(peak_regions)
    newIPI = ipi_proc(IPI)
    #IPI_old = np.concatenate(IPI)
    s_inds = np.sort(np.random.choice(len(IPI)+1, len(peak_regions)))
    shuffled = np.empty(0)
    cursor = 0
    for i, s in enumerate(s_inds):
        p_start, p_end = peak_regions[i]
        shuffled = np.concatenate((shuffled, newIPI[cursor:s],
                                   data_array[p_start:p_end]))
        cursor = s
    if debug:
        return np.concatenate((shuffled, newIPI[cursor:])), points
    else:
        return np.concatenate((shuffled, newIPI[cursor:]))


def background_processing(data_array, perc, debug):
    """
    Process data_array and generates start and end criterion and clean the
    bleaching effect from data
    Input:
        perc: float
            percentile of the gradient, functioning as a cutoff threshold for zeroing
            Higher the value, longer the tail
    Output:
        data_array: ndarray
            cleaned data
        bg: ndarray:
            background signal
        sf: function
            start function that takes in (data_array, i) where i is the
            index that signifies the start of a peak region (inclusive)
        ef: function
            end function that takes in (data_array, j) where j the index
            that signifies the end of a peak region (exclusive),
            therefore, data_array[i:j] would represent one peak region

    """
    grads = np.gradient(data_array)
    bounds = 1 * np.std(grads)
    targets = grads[grads < 0]
    negs_thres = np.percentile(targets, perc) if len(targets) else 0

    def grad_sf(a, i):
        if debug:
            print("Boundary is ", bounds)
        return grads[i] >= bounds

    def in_tail(grads, i):
        return negs_thres <= grads[i] <= 0

    def grad_ef(a, i):
        return i == len(a) - 1 or \
               (grads[i - 1] < 0 and not in_tail(grads, i - 1) and in_tail(
                   grads, i))

    return data_array, np.zeros_like(data_array), grad_sf, grad_ef


def signal_partition(data_array, perc=50, debug=False):
    """ Takes in data_array containing calcium signal and partition them into
    peak regions and IPRIs (Inter Peak Region Intervals, IPI)
    Input:
        data_array: ndarray
            Numpy array with peak data and inter-peak intervals
        perc: float
            percentile of the gradient, functioning as a cutoff threshold for
            zeroing; higher the value, longer the tail
        debug: boolean
            True for debug options
    Output:
        peak_regions: array of tuples
            array of peak region denoted as (start, end)
        IPIs: array of sub-arrays
            array of inter peak region interval arrays
    """
    data_array = data_array[~np.isnan(data_array)]
    data_array, bg, sf, ef = background_processing(data_array, perc, debug)
    peak_regions = []
    prev_end = 0
    IPIs = []
    pk_start = None
    gradients = np.gradient(data_array)

    # TODO: PROCEDURE ONLY APPLIES TO 1D NOW
    for i in range(len(data_array)):
        if debug:
            print("At", i, "Gradient", gradients[i], pk_start, prev_end)
        if pk_start is None:
            if sf(data_array, i):
                # Outside of peak regions, wait for criterion sf to be met
                pk_start = i
                IPIs.append(data_array[prev_end:pk_start])
        else:
            # Within peak_regions, record the peak if ef criterion is met
            if ef(data_array, i):
                peak_regions.append((pk_start, i))
                if debug:
                    print("New peak region", (pk_start, i))
                prev_end = i
                pk_start = None
                if sf(data_array, i):
                    # Outside of peak regions, wait for criterion sf to be met
                    pk_start = i
                    IPIs.append(data_array[prev_end:pk_start])

    buffer = 0
    temp = []

    for i in range(len(peak_regions)):
        if i == 0:
            temp.append(peak_regions[i])
        else:
            start, end = temp[-1]
            curr_start, curr_end = peak_regions[i]
            if curr_start - end <= buffer:
                temp[-1] = (start, curr_end)
            else:
                temp.append(peak_regions[i])
    peak_regions = temp
    if debug:
        points = np.empty(0, dtype=np.int64)
        for pr in peak_regions:
            s, e = pr
            if debug:
                points = np.concatenate((points, np.arange(s, e)))

    IPIs.append(data_array[prev_end:])
    if debug:
        return peak_regions, IPIs, points
    else:
        return peak_regions, IPIs