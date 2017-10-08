import pandas as pd
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]
df = pd.read_csv(filename,index_col=["ID"])
ax = df.plot()
ax.set_xlabel("Data_ID")
ax.set_ylabel("load_value")
plt.show()
