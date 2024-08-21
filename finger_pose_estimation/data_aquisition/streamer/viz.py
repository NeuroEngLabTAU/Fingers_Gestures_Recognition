import warnings
import sklearn.exceptions
from sklearn.decomposition import FastICA
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import scipy.signal as sig
from PIL import ImageGrab
import numpy as np

from .data import Data, ConnectionTimeoutError

warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Viz:

    def __init__(self,
                 data: Data,
                 window_secs: float = 10,
                 plot_exg: bool = True,
                 plot_imu: bool = True,
                 plot_ica: bool = True,
                 ylim_exg: tuple = (-100, 100),
                 ylim_imu: tuple = (-1, 1),
                 update_interval_ms: int = 200,
                 max_points: (int, None) = 1000,
                 max_timeout: (int, None) = 15,
                 find_emg: bool = False,
                 filters: dict = None):

        assert plot_exg or plot_imu

        self.data = data
        self.plot_exg = plot_exg
        self.plot_imu = plot_imu
        self.plot_ica = plot_ica if plot_ica and plot_exg else False
        self.window_secs = window_secs
        self.axes = None
        self.figure = None
        self.ylim_exg = ylim_exg
        self.ylim_imu = ylim_imu
        self.xdata = None
        self.ydata = None
        self.lines = []
        self.bg = None
        self.last_exg_sample = 0
        self.last_imu_sample = 0
        self.init_time = datetime.now()
        self.update_interval_ms = update_interval_ms
        self.max_points = max_points
        self.find_emg = find_emg
        self.filters = filters
        self._backend = None
        self._existing_patches = []
        self.spectrogram_fig = None
        self.spectrogram_axes = None
        self.spectrogram_images = []

        # Confirm initial data retrieval before continuing (or raise Error if timeout)
        while not (self.data.is_connected and self.data.has_data):
            plt.pause(0.01)
            if max_timeout is not None and (datetime.now() - self.init_time).seconds > max_timeout:
                if not data.is_connected:
                    raise ConnectionTimeoutError
                elif not data.has_data:
                    raise TimeoutError(f"Did not succeed to stream data within {max_timeout} seconds.")

        self.setup()

    def setup(self):

        self.ylim_imu = self.ylim_imu if self.ylim_imu else self.ylim_imu
        self.ylim_exg = self.ylim_exg if self.ylim_exg else self.ylim_exg

        # Get data
        n_exg_samples, n_exg_channels = self.data.exg_data.shape if self.plot_exg else (0, 0)
        n_imu_samples, n_imu_channels = self.data.imu_data.shape if self.plot_imu else (0, 0)
        fs = self.data.fs_exg if self.plot_exg else self.data.fs_imu

        # Make timestamp vector
        max_samples = max((n_imu_samples, n_exg_samples))
        last_sec = max_samples / fs
        ts_max = self.window_secs if max_samples <= self.window_secs*fs else last_sec
        ts_min = ts_max - self.window_secs
        ts = np.arange(ts_min, ts_max, 1/fs)
        if self.max_points is None:
            self.max_points = len(ts)
        else:
            ts = Viz._downsample_monotonic(ts, n_pts=self.max_points)

        n_channels = n_exg_channels + n_imu_channels
        self.xdata = ts
        self.ydata = np.full((len(ts), n_channels), np.nan)

        # For auto-maximization of figure
        screensize = ImageGrab.grab().size
        px = 1 / plt.rcParams['figure.dpi']
        screensize_inches = [px*npx for npx in screensize]

        # Prepare plots
        n_rows = n_exg_channels + 1*self.plot_imu
        n_cols = 2 if self.plot_ica else 1
        f, axs = plt.subplots(n_rows, n_cols, sharex=True, figsize=screensize_inches)
        axs = np.atleast_2d(axs).T if n_cols == 1 else axs
        plt.subplots_adjust(hspace=0, left=0.07, right=0.98, top=0.96, bottom=0.04)
        for col in range(n_cols):
            for row in range(n_exg_channels):
                ax = axs[row, col]
                line, = ax.plot(self.xdata, self.ydata[:, row], linewidth=0.5)
                ax.set_ylim((-1, 1)) if col == 1 else axs[row, col].set_ylim(self.ylim_exg)
                self.lines.append(line)
                ax.set_ylabel(f"{'IC' if col == 1 else 'col'} {row+1 if col == 1 else row}", fontsize=11-np.sqrt(n_rows))
                ax.xaxis.set_ticklabels([])
            for n, ch in enumerate(range(n_imu_channels)):
                line, = axs[-1, col].plot(self.xdata, self.ydata[:, n_exg_channels + n], linewidth=0.5)
                self.lines.append(line)
                axs[-1, col].set_ylabel("IMU") if col == 0 else None
                axs[-1, col].set_ylim(*self.ylim_imu)
        axs[0, 0].set_xlim((ts[0], ts[-1]))
        [axs[row, col].spines['bottom'].set_visible(False) for row in range(n_rows-1) for col in range(n_cols)]
        [axs[row, col].spines['top'].set_visible(False) for row in range(1, n_rows) for col in range(n_cols)]
        axs[0, 0].set_title("Raw data")
        axs[0, 1].set_title("ICA") if self.plot_ica else None

        for ax in axs[-1, :]:
            ax.tick_params(axis='x', bottom=False, top=False, labelbottom=True)

        self.axes = axs
        self.figure = f
        self.figure.canvas.mpl_connect('close_event', self.close)

        self.bg = [self.figure.canvas.copy_from_bbox(ax.bbox) for ax in np.ravel(self.axes)] if self._backend != 'module://mplopengl.backend_qtgl' else None

        # Prepare spectrogram window
        if self.plot_exg:
            self.spectrogram_fig, self.spectrogram_axes = plt.subplots(n_exg_channels, 1, figsize=(10, 2 * n_exg_channels))
            self.spectrogram_axes = np.atleast_2d(self.spectrogram_axes).T
            for ax in self.spectrogram_axes:
                img = ax[0].imshow(np.zeros((128, 128)), aspect='auto', origin='lower', cmap='viridis')
                self.spectrogram_images.append(img)

    @staticmethod
    def _downsample_monotonic(arr: np.ndarray, n_pts: int):
        out = arr[[int(np.round(idx)) for idx in np.linspace(0, len(arr)-1, n_pts)]]
        return out

    @staticmethod
    def _correct_matrix(matrix, desired_samples):
        if matrix.shape[0] < desired_samples:
            n_to_pad = desired_samples - matrix.shape[0]
            matrix = np.pad(matrix.astype(float), ((0, n_to_pad), (0, 0)), mode='constant', constant_values=np.nan)
        return matrix

    @staticmethod
    def _crop(matrix, desired_samples):
        nsamples = matrix.shape[0]
        n_samples_cropped = nsamples - desired_samples
        data = matrix[n_samples_cropped:]
        return data, n_samples_cropped

    def _update_data(self, data: Data):

        # Get new data
        if self.plot_exg and data.exg_data is not None:
            n_exg_samples, n_exg_channels = data.exg_data.shape
            new_data_exg = data.exg_data[self.last_exg_sample:, :]
            self.last_exg_sample = n_exg_samples
        else:
            new_data_exg = []
        if self.plot_imu and data.imu_data is not None:
            n_imu_samples, n_imu_channels = data.imu_data.shape
            new_data_imu = data.imu_data[self.last_imu_sample:, :]
            self.last_imu_sample = n_imu_samples
        else:
            new_data_imu = []

        if self.plot_exg and self.plot_imu:
            q = data.fs_imu / data.fs_exg
            assert q - int(q) == 0
            q = int(q)
            new_data_imu = sig.resample_poly(new_data_imu, up=1, down=q)
            fs = data.fs_exg
        elif self.plot_exg:
            fs = data.fs_exg
        elif self.plot_imu:
            fs = data.fs_imu
        else:
            raise RuntimeError

        # Get old data
        old_data_exg = np.full((0, n_exg_channels), np.nan) if self.ydata is None else self.ydata[:, :n_exg_channels]
        old_data_imu = np.full((0, n_imu_channels), np.nan) if self.ydata is None else self.ydata[:, n_exg_channels:]

        data_exg = np.vstack((old_data_exg, new_data_exg)) if old_data_exg.size else new_data_exg
        data_imu = np.vstack((old_data_imu, new_data_imu)) if old_data_imu.size else new_data_imu

        if self.plot_exg and self.plot_imu:
            max_len = max(data_exg.shape[0], data_imu.shape[0])
            data_exg = Viz._correct_matrix(data_exg, max_len)
            data_imu = Viz._correct_matrix(data_imu, max_len)
            all_data = np.hstack((data_exg, data_imu))

        elif self.plot_exg:
            data_exg = Viz._correct_matrix(data_exg, data_exg.shape[0])
            all_data = data_exg

        elif self.plot_imu:
            data_imu = Viz._correct_matrix(data_imu, data_imu.shape[0])
            all_data = data_imu
        else:
            raise RuntimeError

        desired_samples = int(self.window_secs * fs)
        all_data = Viz._correct_matrix(all_data, desired_samples)
        all_data, n_samples_cropped = Viz._crop(all_data, desired_samples)

        if self.xdata is None:
            max_samples = max((n_imu_samples, n_exg_samples))
            last_sec = max_samples / fs
            ts_max = self.window_secs if max_samples <= self.window_secs * fs else last_sec
            ts_min = ts_max - self.window_secs
            self.xdata = np.arange(ts_min, ts_max, 1/fs)

        self.xdata += n_samples_cropped / fs
        self.ydata = all_data

    @staticmethod
    def _format_time(time):
        time_str = str(time)
        if '.' in time_str:
            time_str, ms = time_str.split('.')
            time_str = f'{time_str}.{ms[0]}'
        return time_str

    @staticmethod
    def _nandecimate(data: np.ndarray, n_pts: int):
        if data.shape[0] <= n_pts:
            return data

        non_nan_idc = np.where(~np.all(np.isnan(data), axis=1))[0]
        data_nonan = data[non_nan_idc, :]
        proportion_nonan = data_nonan.shape[0] / data.shape[0]
        n_pts_resample = int(np.round(n_pts * proportion_nonan))
        resampled = sig.resample(data_nonan, n_pts_resample, axis=0)
        out = np.full((n_pts, data.shape[1]), np.nan)
        non_nan_idc_downsampled = np.array([int(i * n_pts / data.shape[0]) for i in
                                            non_nan_idc[np.array([int(np.round(idx)) for idx in
                                                                  np.linspace(0, len(non_nan_idc) - 1, num=n_pts_resample)])]])

        if n_pts in non_nan_idc_downsampled:
            nan_idc_downsampled = [idx for idx in np.arange(0, n_pts) if idx not in non_nan_idc_downsampled]
            last_nan_idx = nan_idc_downsampled[-1]
            non_nan_idc_downsampled = np.array([idx if idx < last_nan_idx else idx - 1 for idx in non_nan_idc_downsampled])

        out[non_nan_idc_downsampled, :] = resampled

        return out

    def update(self, *args, **kwargs):

        self._update_data(self.data)

        n_exg_channels = self.data.exg_data.shape[1] if self.plot_exg else 0
        n_imu_channels = self.data.imu_data.shape[1] if self.plot_imu else 0
        q = int(len(self.xdata)/self.max_points)
        n_pts = int(len(self.xdata) / q)
        x = sig.decimate(self.xdata, q)
        y = Viz._nandecimate(self.ydata, n_pts)
        for n in range(n_exg_channels + n_imu_channels):
            self.lines[n].set_data(x, y[:, n])
        self.axes[-1, -1].set_xlim((x[0], x[-1]))

        duration = datetime.now() - self.init_time
        time_right = Viz._format_time(duration)
        time_left = max(timedelta(seconds=0), duration - timedelta(seconds=self.window_secs))
        time_left = Viz._format_time(time_left)
        time_txt_artists = []
        for ax in self.axes[-1, :]:
            time_txt_artists.append(ax.text(0, 0.05, time_left, transform=ax.transAxes, ha="left", size=9))
            time_txt_artists.append(ax.text(0, 0.05, time_right, transform=ax.transAxes, ha="right", size=9))

        if self.find_emg:
            from viz_emg import plot_emg
            patch_artists = plot_emg(self.axes[:n_exg_channels], x, y, self.data.fs_exg)
        else:
            patch_artists = []

        if self.plot_ica:
            non_nan_idc = np.where(~np.all(np.isnan(self.ydata[:, :n_exg_channels]), axis=1))[0]
            y = self.ydata[non_nan_idc, :n_exg_channels]
            if min(y.shape) == n_exg_channels:
                solver = "eigh" if y.shape[0] >= 50*y.shape[1] else "svd"
                ica = FastICA(whiten_solver=solver)
                ics_nonan = ica.fit_transform(y[:, :n_exg_channels])
                ics = np.full_like(self.ydata[:, :n_exg_channels], np.nan)
                ics[non_nan_idc, :] = ics_nonan
                ics_decim = Viz._nandecimate(ics, n_pts)
                for n in range(n_exg_channels):
                    self.lines[n_exg_channels + n_imu_channels + n].set_data(x, ics_decim[:, n])
            else:
                pass

        import matplotlib.dates as mdates
        myFmt = mdates.DateFormatter('%H:%M:%S')
        [ax.xaxis.set_major_formatter(myFmt) for ax in self.axes[-1, :]]
        [ax.tick_params(axis="x", direction="in", pad=-15, size=9) for ax in self.axes[-1, :]]
        [ax.axes.xaxis.set_ticklabels([]) for ax in self.axes[-1, :]]

        self._update_spectrograms(y, n_exg_channels)

        lines_artists = self.lines
        xtick_artists = []
        for ax in self.axes[-1, :]:
            for artist in ax.get_xticklabels():
                artist.axes = ax
                xtick_artists.append(artist)
        artists = lines_artists + time_txt_artists + xtick_artists + patch_artists

        return artists

    def _update_spectrograms(self, y, n_exg_channels):
        for i in range(n_exg_channels):
            f, t, Sxx = sig.spectrogram(y[:, i], fs=self.data.fs_exg)
            self.spectrogram_images[i].set_data(10 * np.log10(Sxx + 1e-7))
            self.spectrogram_axes[i, 0].set_ylabel(f'Channel {i+1}')
        self.spectrogram_fig.canvas.draw()

    def close(self, _):
        self.data.is_connected = False
        print('Window closed.')

    def start(self):

        from matplotlib.animation import FuncAnimation
        animation = FuncAnimation(self.figure, self.update,
                                  blit=True, interval=self.update_interval_ms, repeat=False, cache_frame_data=False)
        plt.show()
