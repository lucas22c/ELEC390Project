import pandas as pd
import numpy as np

dataset = pd.DataFrame([])

dftemp = pd.read_csv('HaydenMurphyJumpingData.csv')
dataset = pd.concat([dataset, dftemp])

dftemp = pd.read_csv('HaydenMurphyWalkingData.csv')
dataset = pd.concat([dataset, dftemp])

dftemp = pd.read_csv('JacobHicklingJumpingData.csv')
dataset = pd.concat([dataset, dftemp])

dftemp = pd.read_csv('HaydenMurphyWalkingData.csv')
dataset = pd.concat([dataset, dftemp])

dftemp = pd.read_csv('LucasCosterJumpingData.csv')
dataset = pd.concat([dataset, dftemp])

dftemp = pd.read_csv('LucasCosterWalkingData.csv')
dataset = pd.concat([dataset, dftemp])

dataset.to_csv('combinedData.csv')