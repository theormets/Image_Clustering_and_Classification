from google.colab import drive

drive.mount("/content/drive")

import pandas as pd
from matplotlib import pyplot as plt
from pylab import rcParams

rcParams["figure.figsize"] = 7, 7
import seaborn as sns
import numpy as np

sns.set(color_codes=True, font_scale=1.2)

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
%load_ext autoreload
%autoreload 2


xls = pd.ExcelFile("/content/drive/Histogram.xlsx")
df1 = pd.read_excel(xls, "Temp")
df2 = pd.read_excel(xls, "Type")
df = df2
df["KLD"] = df["KLD"].round(2)

import matplotlib

matplotlib.pyplot.switch_backend("agg")
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

heatmap_data = df.pivot_table(values="KLD", index="Group", columns="VS")
cm = ["Blues", "Reds", "Greens"]
num_rows = len(heatmap_data.index)
f, axs = plt.subplots(num_rows, 1, figsize=(16, 12), gridspec_kw={"hspace": 0})
counter = 0
for index, row in heatmap_data.iterrows():
    data = np.array([row.values])
    sns.heatmap(
        data,
        yticklabels=[index],
        xticklabels=heatmap_data.columns,
        annot=True,
        fmt=".2f",
        ax=axs[counter],
        cmap=cm[counter],
        cbar=False,
    )
    counter += 1
plt.suptitle("Heatmaps of KLD Values with Different Colormaps")  # Common title
plt.xlabel("VS")
plt.ylabel("Group")
plt.subplots_adjust(left=0.2)
plt.show()

import plotly.graph_objects as go
import datetime
import numpy as np

np.random.seed(1)
fig = go.Figure(
    data=go.Heatmap(z=df["KLD"], x=df["Group"], y=df["VS"], colorscale="Viridis")
)
fig.update_layout(title="GitHub commits per day", xaxis_nticks=36)
fig.show()

heatmap_data = df["KLD"]
fig = go.Figure(
    data=go.Heatmap(
        z=heatmap_data,
        x=df["VS"],
        y=df["Group"],
        hoverongaps=False,
        text=heatmap_data.values,
        hoverinfo="text",
    )
)
fig = go.Figure(data=fig)
fig.update_layout(
    title="Heatmap of KLD Values", xaxis=dict(title="VS"), yaxis=dict(title="Group")
)
fig.show()

heatmap_data = df_AR.pivot("Group", "VS", "KLD")
plt.imshow(heatmap_data, cmap="hot", interpolation="nearest")
plt.colorbar()
plt.xticks(np.arange(len(heatmap_data.columns)), heatmap_data.columns)
plt.yticks(np.arange(len(heatmap_data.index)), heatmap_data.index)
plt.xlabel("VS")
plt.ylabel("KLD")
plt.title("KLD Heatmap")
plt.show()

heatmap_data = df.pivot_table(values="KLD", index="Group", columns="VS")
plt.figure(figsize=(15, 10))  # Adjust the width and height as needed
sns.heatmap(heatmap_data, annot=True, cmap="coolwarm")
plt.title("Heatmap of KLD Values")
plt.xlabel("VS")
plt.ylabel("Group")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()
