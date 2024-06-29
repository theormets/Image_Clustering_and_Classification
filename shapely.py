
from shapely.geometry import Polygon, mapping
import matplotlib.pyplot as plt
from descartes import PolygonPatch

import pandas as pd
df = pd.read_csv('/content/drive/dataset.csv', index_col=0)
df.tail(5)

df2 = df.groupby(['image_url']).count()

import re
import numpy as np
df2.shape

count_images = df2.shape[0]
count_images

polyList = []
i = 0
for x in range(count_images):
    for y in range(df2.point[x]):
        try:
            points = df.iloc[i].polygon
            i = i + 1
            points_split = re.sub('[()]', '', points)
            list = points_split[1:-1].split(', ')
            arr = np.array(list)
            length = int(len(arr)/2)
            arr2d = np.reshape(arr, (length, 2))
            poly = Polygon(arr2d)
            polyList.append(poly)
        except Exception as k:
            print(i,"Exception:",k)

from PIL import Image
from PIL import ImageDraw
import os
import math

lower = 0
upper = 0

for row in df2.index:
    try:
        im = Image.open("/content/drive/PNG/"+ row)
    except Exception as k:
        print("Exception:",k)

    back_poly = Image.new('RGB', im.size)
    back_poly.paste(im)

    lower = upper
    upper = upper + df2['polygon'][row]

    for c in range(lower, upper):

        poly = Image.new('RGBA', im.size)
        pdraw = ImageDraw.Draw(poly)
        pdraw.polygon(eval(df.iloc[c].polygon),
          fill=(180,50,100,127),outline=(255,0,0,255))

        pdraw.point(eval(df["point"][c]), fill=(255,255,255,255))

        back_poly.paste(poly, (0,0), mask=poly)


    if not os.path.exists("/content/drive"):
        os.makedirs("/content/drive")

    back_poly.save("/content/drive"+ row)
