#from includes.util import sanitize
import numpy as np
import os

class Matrix_loader():
    '''
    meant to extract perfusion parameters from 13x13 CTC data matrixes
    '''
    # Available metrics
    # format of met tuples are (metric, index into b matrix)
    # all is a special case that uses all of them

    def __init__(self, path='.', mat_name='10k_13x13_flair'):
        '''
        path - path to matrices
        mat_name - name of matrixes
        '''
        self.path = path
        self.mat_name = mat_name

    def load_raw_mats(self):
        '''
        loads and returns a tuple of 4 matrices for experiment name
        returns (training data matrix, training target matrix, cross verification data matrix, cross verification target matrix)
        these are the raw 13x13 patch size matrices
        '''
        A_file = os.path.join(self.path, self.mat_name+'_train_A_mmap')
        b_file = os.path.join(self.path, self.mat_name+'_train_b_mmap')

        trainA = np.memmap(A_file, dtype='float64', mode='r')
        trainb = np.memmap(b_file, dtype='float64', mode='r')
        trainA = np.memmap(A_file, dtype='float64', mode='r',\
                                shape=(trainb.shape[0], int(trainA.shape[0]/trainb.shape[0])))

        A_file = os.path.join(self.path, self.mat_name+'_ver_A_mmap')
        b_file = os.path.join(self.path, self.mat_name+'_ver_b_mmap')

        verA = np.memmap(A_file, dtype='float64', mode='r')
        verb = np.memmap(b_file, dtype='float64', mode='r')
        verA = np.memmap(A_file, dtype='float64', mode='r',\
                                shape=(verb.shape[0], int(verA.shape[0]/verb.shape[0])))
        return (trainA, trainb, verA, verb)

    def extract_patches(self, A, halfwindow):

        PSIZE = 13 # patchsize
        AIF_LEN = 40

        temp, aif  = A[:,:-AIF_LEN], A[:,-AIF_LEN:]
        s = temp.shape[1]/(PSIZE**2)
        if np.mod(s,1) != 0:
            print('tried to extract patches with wrong datalen, was given shape %s\n' %repr(A.shape))
            quit()
        temp = np.reshape(temp, (temp.shape[0], PSIZE, PSIZE, int(s)))
        left_l = PSIZE//2 - halfwindow
        right_l = PSIZE//2 + halfwindow + 1
        tA = temp[:, left_l:right_l, left_l:right_l, :]
        tA = np.reshape(tA,(tA.shape[0], -1))
        A = np.hstack((tA, aif))
        return np.matrix(A)

    def load(self, patch_radius):
        '''
        param - str - one of the strings in Matrix_loader.mets
        patch_radius - the radius of the desired patch size
                        patch edge length = 2*patch_radius + 1
                        (patches are square)
                        (CTCs taken from a 13x13 patch were concatenated
                         and then concatenated with AIF to form feature vectors)
        '''

        tA, tb, vA, vb = self.load_raw_mats()

        tA2 = self.extract_patches(tA, patch_radius)
        vA2 = self.extract_patches(vA, patch_radius)

        tb2 = np.matrix(tb).T
        vb2 = np.matrix(vb).T

        return tA2, tb2, vA2, vb2

m = Matrix_loader()
