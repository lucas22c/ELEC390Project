import h5py
import pandas as pd
from sklearn.model_selection import train_test_split


def segment_data(data, window_size, overlap):
    segments = []
    for i in range(0, len(data) - window_size, overlap):
        segments.append(data[i:i + window_size])
    return segments


HaydenWalking = pd.read_csv('HaydenMurphyWalkingData.csv')
HaydenJumping = pd.read_csv('HaydenMurphyJumpingData.csv')
JacobWalking = pd.read_csv('JacobHicklingWalkingData.csv')
JacobJumping = pd.read_csv('JacobHicklingJumpingData.csv')
LucasWalking = pd.read_csv('LucasCosterWalkingData.csv')
LucasJumping = pd.read_csv('LucasCosterJumpingData.csv')

window_size = 5 * 100
overlap = window_size // 2

HaydenWalkingSegments = segment_data(HaydenWalking.values, window_size, overlap)
HaydenJumpingSegments = segment_data(HaydenJumping.values, window_size, overlap)
JacobWalkingSegments = segment_data(JacobWalking.values, window_size, overlap)
JacobJumpingSegments = segment_data(JacobJumping.values, window_size, overlap)
LucasWalkingSegments = segment_data(LucasWalking.values, window_size, overlap)
LucasJumpingSegments = segment_data(LucasJumping.values, window_size, overlap)

with h5py.File('./project_data.h5', 'w') as hdf:
    for name, walkingsegment, jumpingsegment in [('Hayden', HaydenWalkingSegments, HaydenJumpingSegments),
                                                 ('Jacob', JacobWalkingSegments, JacobJumpingSegments),
                                                 ('Lucas', LucasWalkingSegments, LucasJumpingSegments)]:
        person_group = hdf.create_group(name)
        person_group.create_dataset('walking', data=walkingsegment)
        person_group.create_dataset('jumping', data=jumpingsegment)

        walkingtrain, walkingtest = train_test_split(walkingsegment, test_size=0.1, random_state=42)
        jumpingtrain, jumpingtest = train_test_split(jumpingsegment, test_size=0.1, random_state=42)

        traingroup = hdf.create_group(f'{name}/Train')
        traingroup.create_dataset('walking', data=walkingtrain)
        traingroup.create_dataset('jumping', data=jumpingtrain)

        testgroup = hdf.create_group(f'{name}/Test')
        testgroup.create_dataset('walking', data=walkingtest)
        testgroup.create_dataset('jumping', data=jumpingtest)
    hdf.close()