import transformer as tfr 

import matplotlib.pyplot as plt 

def test1():
    """subsequent_mask"""
    # plt.figure(figsize=(5,5))
    # plt.imshow(tfr.subsequent_mask(20)[0])
    plt.imsave('test_image.jpg', tfr.subsequent_mask(20)[0])
    print(tfr.subsequent_mask(20)[0])
    print('done')

















test1()