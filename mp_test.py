import multiprocessing as mp

# test multi process writing to the same txt file and use cat to test consistency
# Use ipython to see if it works
count = 0

def little(f):
    global count
    count += 1
    print(f, count)

print(mp.Pool(3).map_async(little, range(10), chunksize=4).get())

