import os
import csv
from pathlib import Path


for filename in os.popen('ls *.csv').read().splitlines():
    xdata = []
    ydata = []
    with open(Path(filename), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            xdata.append(float(row[0]))
            ydata.append(float(row[1]))
    with open(Path(str(Path(filename).stem)+'_pathological.csv'), 'w') as f:
        writer = csv.writer(f)
        for i in range(len(xdata)):
            writer.writerow([xdata[i]+45, ydata[i]])
