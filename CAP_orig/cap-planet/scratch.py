import plotly
import chart_studio
# plotly.tools.set_credentials_file(username='lathkar', api_key='********************')
import chart_studio.plotly as py
import plotly.graph_objs as go
import numpy as np
import math #needed for definition of pi
import os

path = "./"

xpoints = np.arange(0, math.pi*2, 0.05)
ypoints = np.sin(xpoints)
trace0 = go.Scatter(
   x = xpoints, y = ypoints
)
data = [trace0]
# py.plot(data, filename = 'Sine wave', auto_open=True)


plotly.offline.plot(
   { "data": data,"layout": go.Layout(title = "hello world")}, filename=os.path.join(path, "hello" + '.html'), auto_open=False)
