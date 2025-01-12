# 导入原始数据
import numpy as np
import mne
import os
import gdown
import zipfile
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
plt.rcParams["figure.dpi"] = 150

import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [4, 5, 6]

plt.plot(x, y)
plt.title('Sample Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('plot.png')
plt.show()
data_dir = "data/"
if not os.path.exists(data_dir):
  os.makedirs(data_dir)
url = "https://drive.google.com/file/d/1bXD-_dDnH5Mv3DQrV7V9fYM4-xYsZ0DN/view?usp=sharing"
filename = "sample_data"
filepath = data_dir + filename + ".zip"
gdown.download(url=url, output=filepath, quiet=False, fuzzy=True)
print("Download completes!")
with zipfile.ZipFile(filepath, 'r') as zip:
    zip.extractall(data_dir)
print("Unzip completes!")
data_path = data_dir + 'sample_data/eeglab_data.set'
raw = mne.io.read_raw_eeglab(data_path, preload=True)
#查看原始数据
print(raw)
print(raw.info)
#电极定位
locs_info_path = data_dir + "sample_data/eeglab_chan32.locs"
montage = mne.channels.read_custom_montage(locs_info_path)
new_chan_names = np.loadtxt(locs_info_path, dtype=str, usecols=3)
old_chan_names = raw.info["ch_names"]
chan_names_dict = {old_chan_names[i]:new_chan_names[i] for i in range(32)}
raw.rename_channels(chan_names_dict)
raw.set_montage(montage)
#设置导联类型
chan_types_dict = {"EOG1":"eog", "EOG2":"eog"}
raw.set_channel_types(chan_types_dict)
chan_types_dict = {"EOG1":"eog", "EOG2":"eog"}
raw.set_channel_types(chan_types_dict)
#查看修改后的数据信息
print(raw.info)
#绘制原始数据波形图
raw.plot(duration=5, n_channels=32, clipping=None)
#绘制原始数据功率谱图
raw.plot_psd(average=True)
#绘制导联空间位置图
raw.plot_sensors(ch_type='eeg', show_names=True)
#绘制拓扑图形式的原始数据功率谱图
raw.compute_psd().plot_topo()
#陷波滤波
raw = raw.notch_filter(freqs=(60))
raw.plot_psd(average=True)
#高/低通滤波
raw = raw.filter(l_freq=0.1, h_freq=30)
raw.plot_psd(average=True)
#去伪迹
fig = raw.plot()
fig.fake_keypress('a')
raw.info['bads'].append('FC5')
print(raw.info['bads'])
raw = raw.interpolate_bads()
#独立成分分析
ica = ICA(max_iter='auto')
raw_for_ica = raw.copy().filter(l_freq=1, h_freq=None)
ica.fit(raw_for_ica)
ica.plot_sources(raw_for_ica)
ica.plot_components()
ica.plot_overlay(raw_for_ica, exclude=[1])
ica.plot_properties(raw, picks=[1])
ica.exclude = [1]
ica.exclude = [1]
raw.plot(duration=5, n_channels=32, clipping=None)
#重参考
#数据分段
print(raw.annotations)
print(raw.annotations.duration)
print(raw.annotations.description)
print(raw.annotations.onset)
events, event_id = mne.events_from_annotations(raw)
print(events.shape, event_id)
epochs = mne.Epochs(raw, events, event_id=2, tmin=-1, tmax=2, baseline=(-0.5, 0),
                    preload=True, reject=dict(eeg=2e-4))
print(epochs)
epochs.plot(n_epochs=4)
epochs.compute_psd().plot(picks='eeg')
bands = [(4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta')]
epochs.plot_psd_topomap(bands=bands, vlim='joint')
# 叠加平均
evoked = epochs.average()
evoked.plot()
times = np.linspace(0, 2, 5)
evoked.plot_topomap(times=times, colorbar=True)
evoked.plot_topomap(times=0.8, average=0.1)
evoked.plot_joint()
evoked.plot_image()
evoked.plot_topo()
mne.viz.plot_compare_evokeds(evokeds=evoked, combine='mean')
mne.viz.plot_compare_evokeds(evokeds=evoked, picks=['O1', 'Oz', 'O2'], combine='mean')
#时频分析
freqs = np.logspace(*np.log10([4, 30]), num=10)
n_cycles = freqs / 2.
power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True)
power.plot(picks=['O1', 'Oz', 'O2'], baseline=(-0.5, 0), mode='logratio', title='auto')
power.plot(picks=['O1', 'Oz', 'O2'], baseline=(-0.5, 0), mode='logratio',
           title='Occipital', combine='mean')
power.plot_topo(baseline=(-0.5, 0), mode='logratio', title='Average power')
power.plot_topomap(tmin=0, tmax=0.5, fmin=4, fmax=8,
                   baseline=(-0.5, 0), mode='logratio')
power.plot_topomap(tmin=0, tmax=0.5, fmin=8, fmax=12,
                   baseline=(-0.5, 0), mode='logratio')
power.plot_joint(baseline=(-0.5, 0), mode='mean', tmin=-0.5, tmax=1.5,
                 timefreqs=[(0.5, 10), (1, 8)])
itc.plot_topo(baseline=(-0.5, 0), mode='logratio', title='Average Inter-Trial coherence')
#数据提取
epochs_array = epochs.get_data()
print(epochs_array.shape)
print(epochs_array)
power_array = power.data
print(power_array.shape)
print(power_array)
# step1-eeg
