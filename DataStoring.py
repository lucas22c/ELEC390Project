import pandas as pd
import numpy as np
import h5py

firstMemberWalking = pd.read_csv('HaydenMurphyWalkingData.csv')

firstMemberJumping = pd.read_csv('HaydenMurphyJumpingData.csv')

secondMemberWalking = pd.read_csv('JacobHicklingWalkingData.csv')

secondMemberJumping = pd.read_csv('JacobHicklingJumpingData.csv')

thirdMemberWalking = pd.read_csv('LucasCosterWalkingData.csv')

thirdMemberJumping = pd.read_csv('LucasCosterJumpingData.csv')

walkingJoined = pd.concat([firstMemberWalking, secondMemberWalking, thirdMemberWalking], ignore_index=True)
jumpingJoined = pd.concat([firstMemberJumping, secondMemberJumping, thirdMemberJumping], ignore_index=True)
allJoined = pd.concat([walkingJoined, jumpingJoined], ignore_index=True)

firstMemberWalking = firstMemberWalking.to_numpy()
secondMemberWalking = secondMemberWalking.to_numpy()
thirdMemberWalking = thirdMemberWalking.to_numpy()

firstMemberJumping = firstMemberJumping.to_numpy()
secondMemberJumping = secondMemberJumping.to_numpy()
thirdMemberJumping = thirdMemberJumping.to_numpy()
walkingJoined = walkingJoined.to_numpy()
jumpingJoined = jumpingJoined.to_numpy()
allJoined = allJoined.to_numpy()

with h5py.File('./project_data.h5', 'w') as hdf:
    Hayden = hdf.create_group('Hayden Murphy')
    Hayden.create_dataset('walking', data=firstMemberWalking)
    Hayden.create_dataset('jumping', data=firstMemberJumping)

    Jacob = hdf.create_group('Jacob Hickling')
    Jacob.create_dataset('walking', data=secondMemberWalking)
    Jacob.create_dataset('jumping', data=secondMemberJumping)

    Lucas = hdf.create_group('Lucas Coster')
    Lucas.create_dataset('walking', data=thirdMemberWalking)
    Lucas.create_dataset('jumping', data=thirdMemberJumping)

    dataset_group = hdf.create_group('dataset')

    num_segments = (len(allJoined) - 500) // 100 + 1
    segments = [allJoined[(i * 100):(i * 100 + 500)] for i in range(num_segments)]

    np.random.shuffle(segments)

    nTrain = int(0.9 * len(segments))
    train_segments = segments[:nTrain]
    test_segments = segments[nTrain:]

    dataset_group.create_dataset('test', data=test_segments)
    dataset_group.create_dataset('train', data=train_segments)

with h5py.File('./project_data.h5', 'r') as hdf:
    ls = list(hdf.keys())
    print(ls)