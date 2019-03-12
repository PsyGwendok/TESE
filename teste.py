import pandas as pd
import numpy as np
import os
import natsort
from natsort import natsorted, ns

path = "C:/Users/Psy/Downloads/Data/Elephants"


mylist = sorted(os.listdir(path))
natsorted(mylist, key=lambda y: y.lower())
print(mylist)

#df = pd.DataFrame({"file" : mylist})

#df.to_csv("submission1.csv", index=False)