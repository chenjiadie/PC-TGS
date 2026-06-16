# -*- coding: utf-8 -*-
"""painter for data
"""
import os
import numpy as np
# import matplotlib.pyplot as plt
import torch

def CoarseA(A):
    index = torch.cat([torch.arange(3*i * 72 + 0, 3*i * 72 + 72) for i in range(0,91//3)])#0-30
    index= index[1::3]
    # A = torch.tensor(A)
    A2=torch.zeros((A.shape))
    for i in index:


        a, s, d, q, w, e, z, x, c = FindNearest9Angle(i)
        # A2[...,i]=torch.max( A[...,s],A[...,a],A[...,d],A[...,w],A[...,x],A[...,e],A[...,q],A[...,z],A[...,c])
        # 选择要比较的通道的索引
        channels_to_compare = [s, a, d, w, x, e, q, z, c]

        # 从 A 中选择对应索引的通道，并沿着最后一个维度（通道维度）取最大值
        max_values, _ = torch.max(torch.stack([A[..., i] for i in channels_to_compare], dim=-1), dim=-1)

        # 将最大值填充回 A 的对应位置
        A2[..., i] = max_values
        # A2[...,i]=(2*A[...,s]+1.5*(A[...,a]+A[...,d]+A[...,w]+A[...,x])+(A[...,e]+A[...,q]+A[...,z]+A[...,c]))/12
    return A2

def FindNearest9Angle(i):
    col = i // 72
    s = i

    a = i - 1 if (i - 1) // 72 == col else 72 * col + 71
    d = i + 1 if (i + 1) // 72 == col else 72 * col + 0
    w = i - 72 if i - 72 > 0 else i - 72 + 6552
    x = i + 72 if i + 72 < 6552 else i + 72 - 6552

    if a - 72 < 0:
        q = a - 72 + 6552
        e = d - 72 + 6552
        z = a + 72
        c = d + 72
    elif a + 72 > 6552:
        q = a - 72
        e = d - 72
        z = a + 6552 + 72
        c = d + 6552 + 72
    else:
        q = a - 72
        e = d - 72
        z = a + 72
        c = d + 72
    return a,s,d,q,w,e,z,x,c
def find_no_zero(test_error,i):
    i=i+1
    if test_error[i]==0.0:
        i=find_no_zero(test_error,i)
    else:
        return i
    return i

def test_x_predict(xx,save_train_mae,save_test_mae):
    tmae0, tmae1, tmae2, tmae3, tmae4 = [], [], [], [], []
    temae0, temae1, temae2, temae3, temae4 = [], [], [], [], []

    x_max = np.array(xx)
    i=-1
    for ii in range(len(x_max)):
        i+=1
        # for i in range(len(save_train_mae)):
        if save_train_mae.numpy().mean(-1)[i]==0.0:
            i = find_no_zero(save_train_mae.numpy().mean(-1),i)
        else:
            if x_max[ii] == 0:
                tmae0.append(save_train_mae.numpy().mean(-1)[i])
                temae0.append(save_test_mae.numpy().mean(-1)[i])

            elif x_max[ii] == 5:
                tmae1.append(save_train_mae.numpy().mean(-1)[i])
                temae1.append(save_test_mae.numpy().mean(-1)[i])
            elif x_max[ii] == 10:
                tmae2.append(save_train_mae.numpy().mean(-1)[i])
                temae2.append(save_test_mae.numpy().mean(-1)[i])
            elif x_max[ii] == 100:

                tmae3.append(save_train_mae.numpy().mean(-1)[i])
                temae3.append(save_test_mae.numpy().mean(-1)[i])
            elif x_max[ii] == -1:
                tmae4.append(save_train_mae.numpy().mean(-1)[i])
                temae4.append(save_test_mae.numpy().mean(-1)[i])

    print(len(x_max[x_max == 0]) / x_max.shape[0], len(x_max[x_max == 5]) / x_max.shape[0], len(x_max[x_max == 10]) / x_max.shape[0],
          len(x_max[x_max == 100]) / x_max.shape[0], len(x_max[x_max == -1]) / x_max.shape[0])
    print(np.mean(tmae0), np.mean(tmae1), np.mean(tmae2), np.mean(tmae3), np.mean(tmae4))
    print(np.mean(temae0), np.mean(temae1), np.mean(temae2), np.mean(temae3), np.mean(temae4))
    print(save_train_mae[save_train_mae!=0].numpy().mean( ),save_test_mae[save_test_mae!=0].numpy().mean( ))

def find_max_A(A,max_number=600):
    #A:[6552,32]
    matric=A.T
    norms = np.linalg.norm(matric, axis=1)
    # 排序并选择最大的900个元素
    max_indices = np.argsort(norms)[-max_number:]
    # 选择最大的900个元素
    return max_indices

def paint_spectrum(spectrum, save_path=None):

    spectrum = spectrum.numpy().reshape(90, 360)
    plt.imsave(save_path, spectrum, cmap='jet')
    spectrum = np.flipud(spectrum)
    # create a polar grid
    r = np.linspace(0, 1, 91) # change this depending on your radial distance
    theta = np.linspace(0, 2.*np.pi, 361)

    r, theta = np.meshgrid(r, theta)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    cax = ax.pcolormesh(theta, r, spectrum.T, cmap='jet', shading='flat')
    ax.axis('off')

    # save the image as a PNG file
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)


def paint_spectrum_compare(pred_spectrum, gt_spectrum, save_path=None):

    # create a polar grid
    r = np.linspace(0, 1, 91) # change this depending on your radial distance
    theta = np.linspace(0, 2.*np.pi, 361)

    r, theta = np.meshgrid(r, theta)

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 6))

    cax1 = axs[0].pcolormesh(theta, r, np.flipud(pred_spectrum).T, cmap='viridis', shading='flat')
    axs[0].axis('off')

    cax2 = axs[1].pcolormesh(theta, r, np.flipud(gt_spectrum).T, cmap='viridis', shading='flat')
    axs[1].axis('off')

    # save the image as a PNG file
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()


def paint_location(loc_path, save_path):


    all_loc = np.loadtxt(os.path.join(loc_path, 'tx_pos.csv'), delimiter=',', skiprows=1)
    train_index = np.loadtxt(os.path.join(loc_path, 'train_index.txt'), dtype=int)
    test_index = np.loadtxt(os.path.join(loc_path, 'test_index.txt'), dtype=int)
    train_loc = all_loc[train_index-1]
    test_loc = all_loc[test_index-1]
    plt.scatter(train_loc[:, 0], train_loc[:, 1], c='b', label='train',s=0.1)
    plt.scatter(test_loc[:, 0], test_loc[:, 1], c='r', label='test',s=0.1)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loc.pdf'), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':

    loc_path = "data/s23/"
    save_path = "data/s23/"
    paint_location(loc_path, save_path)