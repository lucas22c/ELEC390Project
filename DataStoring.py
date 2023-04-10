import numpy as np
import h5py

with h5py.File('./hdf5_data.h5', 'w') as hdf:

    name1 = hdf.create_group('/Lucas Coster')
    name1.create_dataset('Walking', 'LucasCosterWalkingData.csv')
    name1.create_dataset('Jumping', 'LucasCosterJumpingData.csv')

    name2 = hdf.create_group('/Hayden Murphy')
    name1.create_dataset('Walking', 'HaydenMurphyWalkingData.csv')
    name1.create_dataset('Jumping', 'HaydenMurphyJumpingData.csv')

    name3 = hdf.create_group('/Jacob Hickling')
    name1.create_dataset('Walking', 'JacobHicklingWalkingData.csv')
    name1.create_dataset('Jumping', 'JacobHicklingJumpingData.csv')
