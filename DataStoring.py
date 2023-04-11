import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def segment_data(data, window_size, overlap):
    segments = []
    for i in range(0, len(data) - window_size, overlap):
        segments.append(data[i:i + window_size])
    return segments


HaydenWalking = pd.read_csv('HaydenMurphyWalkingData.csv').to_numpy()
HaydenJumping = pd.read_csv('HaydenMurphyJumpingData.csv').to_numpy()
JacobWalking = pd.read_csv('JacobHicklingWalkingData.csv').to_numpy()
JacobJumping = pd.read_csv('JacobHicklingJumpingData.csv').to_numpy()
LucasWalking = np.genfromtxt('LucasCosterWalkingData.csv', delimiter=';')
LucasJumping = np.genfromtxt('LucasCosterJumpingData.csv', delimiter=';')

window_size = 5 * 100
overlap = window_size // 2

HaydenWalkingSegments = segment_data(HaydenWalking, window_size, overlap)
HaydenJumpingSegments = segment_data(HaydenJumping, window_size, overlap)
JacobWalkingSegments = segment_data(JacobWalking, window_size, overlap)
JacobJumpingSegments = segment_data(JacobJumping, window_size, overlap)
LucasWalkingSegments = segment_data(LucasWalking, window_size, overlap)
LucasJumpingSegments = segment_data(LucasJumping, window_size, overlap)

with h5py.File('./project_data.h5', 'w') as hdf:

    for name, walkingsegment, jumpingsegment in [('Hayden', HaydenWalking, HaydenJumping),
                                                 ('Jacob', JacobWalking, JacobJumping),
                                                 ('Lucas', LucasWalking, LucasJumping)]:
        person_group = hdf.create_group(name)
        person_group.create_dataset('walking', data=walkingsegment)
        person_group.create_dataset('jumping', data=jumpingsegment)

    walkingsegment = np.concatenate([HaydenWalkingSegments, JacobWalkingSegments, LucasWalkingSegments])
    jumpingsegment = np.concatenate([HaydenJumpingSegments, JacobJumpingSegments, LucasJumpingSegments])

    walkingtrain, walkingtest = train_test_split(walkingsegment, test_size=0.1, random_state=42)
    jumpingtrain, jumpingtest = train_test_split(jumpingsegment, test_size=0.1, random_state=42)

    train = hdf.create_group('Dataset/Train')
    train.create_dataset('walking', data=walkingtrain)
    train.create_dataset('jumping', data=jumpingtrain)

    test = hdf.create_group('Dataset/Test')
    test.create_dataset('walking', data=walkingtest)
    test.create_dataset('jumping', data=jumpingtest)

    hdf.close()
