import numpy as np
import sounddevice as sd
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg
import time

SAMPLE_RATE = 44100
BUFFER_SIZE = 4096
FFT_SIZE = BUFFER_SIZE
WINDOW = np.hanning(FFT_SIZE)
REFERENCE = 1.0

USE_SYNTHETIC = False

class SpectrumAnalyzer:
    def __init__(self):
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title="spectrum analyzer")
        self.plot = self.win.addPlot(title="frequency spectrum")
        self.plot.setLogMode(x=True, y=False)
        self.plot.setLabel('bottom', 'frequency', units='Hz')
        self.plot.setLabel('left', 'magnitude', units='dB')
        self.curve = self.plot.plot(pen='c')
        self.plot.setYRange(-100, 0)
        self.freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(30)

        self.fft_data_db = np.full(len(self.freqs), -100.0)

        self.use_synthetic = False
        try:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=BUFFER_SIZE,
                callback=self.audio_callback,
                dtype='float32'
            )
            self.stream.start()
        except Exception as e:
            print(f"[Warning] Audio input failed ({e}), falling back to synthetic signal.")
            self.use_synthetic = True
            self.start_time = time.time()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Stream status:", status)
        samples = indata[:, 0]
        self.process_buffer(samples)

    def process_buffer(self, buffer):
        if len(buffer) != FFT_SIZE:
            if len(buffer) < FFT_SIZE:
                buf = np.zeros(FFT_SIZE, dtype='float32')
                buf[:len(buffer)] = buffer
            else:
                buf = buffer[:FFT_SIZE]
        else:
            buf = buffer

        windowed = buf * WINDOW
        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft) / (np.sum(WINDOW) / 2)
        eps = 1e-10
        mag_db = 20 * np.log10(magnitude + eps / REFERENCE)
        self.fft_data_db = mag_db

    def generate_synthetic(self):
        t = np.linspace(0, BUFFER_SIZE / SAMPLE_RATE, BUFFER_SIZE, endpoint=False)
        sig = 0.6 * np.sin(2 * np.pi * 440 * t)
        sig += 0.3 * np.sin(2 * np.pi * 1000 * t)
        sig += 0.1 * np.random.normal(size=BUFFER_SIZE)
        return sig.astype('float32')

    def update_plot(self):
        if self.use_synthetic:
            synthetic = self.generate_synthetic()
            self.process_buffer(synthetic)

        self.curve.setData(self.freqs, self.fft_data_db)

    def run(self):
        self.win.show()
        QtWidgets.QApplication.instance().exec()

if __name__ == "__main__":
    analyzer = SpectrumAnalyzer()
    analyzer.run()