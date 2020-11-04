import transformer as tfr 
import torch
import numpy as np
import matplotlib.pyplot as plt 

def test1():
    """subsequent_mask"""
    # plt.figure(figsize=(5,5))
    # plt.imshow(tfr.subsequent_mask(20)[0])
    plt.imsave('test_image.jpg', tfr.subsequent_mask(20)[0])
    print(tfr.subsequent_mask(20)[0])
    print('done')

def test2():
    plt.figure(figsize=(15, 5))
    pe = tfr.PositionalEncoding(20, 0)
    y = pe.forward(torch.autograd.Variable(torch.zeros(1, 100, 20)))
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    plt.legend(["dim %d"%p for p in [4,5,6,7]])
    plt.savefig('save.jpg')
















# test1()
test2()

