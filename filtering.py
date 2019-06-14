import numpy as np


class DCache:
    # TODO: AUGMENT IT SUCH THAT IT WORKS FOR MULTIPLE

    def __init__(self, size=20, thres=2, buffer=False):
        """
        :param size: int, size of the dampening cache
        :param thres: float, threshold for valid data caching, ignore signal if |x - miu_x| > thres * var
        :param buffer: boolean, for whether keeping a dynamic buffer
        """
        self.size = size
        self.thres = thres
        self.counter = 0
        self.bandwidth = None

        if buffer:
            self.cache = []
        else:
            self.avg = 0
            self.var = 0

    def __len__(self):
        return self.size

    def add(self, signal):
        if self.bandwidth is None:
            self.bandwidth = signal.shape[0]
        if self.counter < self.size:
            #print(self.avg, self.avg * (self.counter - 1), (self.avg * self.counter + signal) / (self.counter + 1))
            self.avg = (self.avg * self.counter + signal) / (self.counter + 1)
            diff2 = (signal - self.avg) ** 2
            self.var = (diff2 + self.var * self.counter) / (self.counter+1)

        else:
            targets = signal - self.avg < np.sqrt(self.var) * self.thres
            #print(self.avg, self.avg * (self.size - 1), (self.avg * (self.size - 1) + signal) / self.size)
            self.avg[targets] = (self.avg[targets] * (self.size - 1) + signal[targets]) / self.size
            diff2 = (signal[targets] - self.avg[targets]) ** 2
            self.var[targets] = (diff2 + self.var[targets] * (self.size - 1)) / self.size
        self.counter += 1

    def get_val(self):
        return self.avg


def std_filter(size=20, thres=2):
    dc = DCache(size, thres)

    def fil(sigs):
        dc.add(sigs)
        #print(sigs[i], dc.get_val())
        return dc.get_val()
    return fil, dc




def fft_filter(sig, start, end):
    ftf = np.fft.fft(sig)
    ftred = np.copy(ftf)
    ftred[:start] = 0
    ftred[end:] = 0
    return abs(np.fft.ifft(ftred))