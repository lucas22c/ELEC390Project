import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# takes a dictionary and segments data accordingly. reworked from old rolling window implementation


def segment_data(dict):

    segments = []

    for person, df in dict.items():
        temp = df["Walking"]
        # get current segment and details, zip together (data was really unintelligible without this)
        current_seg = [temp[(i * 100) : (i * 100 + 500)] for i in range((len(temp) - 500) // 100 + 1)]
        current_details = [f'{person}i{"Walking"}' for i in range((len(temp) - 500) // 100 + 1)]

        segments.extend(list(zip(current_seg, current_details)))

        temp = df["Jumping"]
        # get current segment and details, zip together (data was really unintelligible without this)
        current_seg = [temp[(i * 100): (i * 100 + 500)] for i in range((len(temp) - 500) // 100 + 1)]
        current_details = [f'{person}i{"Jumping"}' for i in range((len(temp) - 500) // 100 + 1)]

        segments.extend(list(zip(current_seg, current_details)))

    return segments

# collect group data
HaydenWalking = pd.read_csv('HaydenMurphyWalkingData.csv')
HaydenJumping = pd.read_csv('HaydenMurphyJumpingData.csv')
JacobWalking = pd.read_csv('JacobHicklingWalkingData.csv')
JacobJumping = pd.read_csv('JacobHicklingJumpingData.csv')
LucasWalking = pd.read_csv('LucasCosterWalkingData.csv', delimiter=';') # just to fix a small error
LucasJumping = pd.read_csv('LucasCosterJumpingData.csv', delimiter=';')

# all group combined and shuffle all members to remove clusters of same member
walking = pd.concat([HaydenWalking, JacobWalking, LucasWalking], ignore_index=True)
walking = pd.DataFrame(np.random.shuffle(walking.to_numpy()))

jumping = pd.concat([HaydenJumping, JacobJumping, LucasJumping], ignore_index=True)
jumping = pd.DataFrame(np.random.shuffle(jumping.to_numpy()))

# for easy storage and segmenting
group_dict = {
    "Hayden" : {"Walking" : HaydenWalking, "Jumping" : HaydenJumping},
    "Jacob" : {"Walking" : JacobWalking, "Jumping" : JacobJumping},
    "Lucas" : {"Walking" : LucasWalking, "Jumping" : LucasJumping}
}

with h5py.File('./project_data.h5', 'w') as f:
    dataset = f.create_group("Dataset")

    segmented_data = segment_data(group_dict)
    np.random.shuffle(segmented_data) # randomize to once again avoid clustering

    train_seg, test_seg = train_test_split(segmented_data, test_size=0.1, shuffle=False)

    train = dataset.create_group("Train")
    train.create_dataset("Walking", data=[df[0] for df in train_seg if "Walking" in df[1]])
    train.create_dataset("Jumping", data=[df[0] for df in train_seg if "Jumping" in df[1]])
    train = dataset.create_group("Test")
    train.create_dataset("Walking", data=[df[0] for df in test_seg if "Walking" in df[1]])
    train.create_dataset("Jumping", data=[df[0] for df in test_seg if "Jumping" in df[1]])

    # store original, unmodified data
    group = f.create_group("Hayden")
    group.create_dataset("Jumping", data=group_dict["Hayden"]["Jumping"])
    group.create_dataset("Walking", data=group_dict["Hayden"]["Walking"])
    group = f.create_group("Jacob")
    group.create_dataset("Jumping", data=group_dict["Jacob"]["Jumping"])
    group.create_dataset("Walking", data=group_dict["Jacob"]["Walking"])
    group = f.create_group("Lucas")
    group.create_dataset("Jumping", data=group_dict["Lucas"]["Jumping"])
    group.create_dataset("Walking", data=group_dict["Lucas"]["Walking"])

    f.close()
