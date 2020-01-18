import numpy as np

def neuron_raw_fano(sig, W=100, step=1, debug=False):
    dis_deconv = sig
    inds = [i for i in range(0, len(sig) - W + 1, step)]
    ms, ss, fanos = [], [], []
    csum, csquare = 0, 0
    for i in inds:
        if i == 0:
            psig = dis_deconv[i:i+W]
            csum, csquare = np.sum(psig), np.sum(np.square(psig))
            m = csum / W
            s = csquare / W - m ** 2
        else:
            csum = csum - sig[i-1] + sig[i+W-1]
            csquare = csquare - sig[i-1] ** 2 + sig[i+W-1] ** 2
            m = csum / W
            s = csquare / W - m ** 2
        if debug:
            print(i, m, s)
        ms.append(m)
        ss.append(s)
        if np.isclose(m, 0) and np.isclose(s, 0):
            fanos.append(1)
        else:
            fanos.append(s/(abs(m)+1e-14))
    return ms, ss, fanos
