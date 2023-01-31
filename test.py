import transformer as tfr 
import torch
from torch.autograd import Variable
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

def test3():
    opts = [tfr.NoamOpt(512, 1, 4000, None), 
        tfr.NoamOpt(512, 1, 8000, None),
        tfr.NoamOpt(256, 1, 4000, None)]
    plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    plt.legend(["512:4000", "512:8000", "256:4000"])
    plt.savefig('save.jpg')



def test4():
    crit = tfr.LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                [0, 0.2, 0.7, 0.1, 0], 
                                [0, 0.2, 0.7, 0.1, 0]])
    v = crit(Variable(predict.log()), 
            Variable(torch.LongTensor([2, 1, 0])))
    
    # Show the target distributions expected by the system.
    plt.imshow(crit.true_dist)
    plt.savefig('save.jpg')









# test1()
test4()

