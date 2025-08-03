import numpy as np
import sounddevice as sd
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg

SAMPLE_RATE = 44100
FFT_SIZE = 4096
WINDOW = np.hanning(FFT_SIZE)
freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE)

class SpectrumAnalyzer:
    def __init__(self):
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title="spectrum analyzer")
        self.plot = self.win.addPlot(title="FFT")
        self.curve = self.plot.plot()
        self.plot.setLabel('bottom', 'frequency', units='Hz')
        self.plot.setLabel('left', 'magnitude (dB)')
        self.plot.setYRange(-100, 0)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(30)
        self.fft_db = np.full(len(freqs), -100.0)

        self.stream = sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=FFT_SIZE,
            callback=self.audio_callback,
            dtype='float32'
        )
        self.stream.start()

    def audio_callback(self, indata, frames, time_info, status):
        samples = indata[:, 0]
        windowed = samples * WINDOW
        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft) / (np.sum(WINDOW) / 2)
        eps = 1e-10
        self.fft_db = 20 * np.log10(magnitude + eps)

    def update_plot(self):
        self.curve.setData(freqs, self.fft_db)

    def run(self):
        self.win.show()
        QtWidgets.QApplication.instance().exec()

if __name__ == "__main__":
    SpectrumAnalyzer().run()
