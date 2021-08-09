"""

"""
import plotly.graph_objects as go
from .import pll_calcs
from .import phase_noise
import numpy as np

def plot_phase_noise(lst_dicts, title=None, num_points=100, freq_comp=None, ylim=[-160,-60], ytick=10):
    """
    """
    traces = []
    if type(lst_dicts) == dict:
        lst_dicts = [lst_dicts]
    for d in lst_dicts:
        if freq_comp != None:
            freq_t, pn_t = phase_noise.translate_phase_noise(d, freq_comp)
        else:
            freq_t = d['offset']
            pn_t = d['phase_noise']
        freq, pn = pll_calcs.interp_semilogx(freq_t, pn_t, num_points=num_points)
        traces.append(go.Scatter(x=freq,
                                 y=pn,
                                 # logx=True,
                                 mode='lines',
                                 name=d['name'],
                                 showlegend=True))
    if title == None:
        title = "Phase Noise"
    fig = go.Figure(data=traces)
    fig.update_layout(title=title, xaxis_type='log')
    fig.update_xaxes(title="offset (Hz)")
    fig.update_yaxes(title="dBc/Hz")
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=np.arange(min(ylim), max(ylim)+ytick, ytick)
        ),
        yaxis_range=(min(ylim), max(ylim))
    )
    fig.show()


