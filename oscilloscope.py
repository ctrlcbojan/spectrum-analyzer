import numpy as np
import sounddevice as sd
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg

SAMPLE_RATE = 44100
BUFFER_SIZE = 1024

class Scope:
    def __init__(self):
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title="oscilloscope")
        self.plot = self.win.addPlot(title="waveform")
        self.curve = self.plot.plot(pen='y')
        self.plot.setYRange(-1, 1)
        self.plot.setLabel('bottom', 'time', 's')
        self.plot.setLabel('left', 'amplitude')
        self.x = np.linspace(0, BUFFER_SIZE / SAMPLE_RATE, BUFFER_SIZE)
        self.data = np.zeros(BUFFER_SIZE)


        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(0)

        self.stream = sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=BUFFER_SIZE,
            callback=self.audio_callback,
            dtype='float32'
        )
        self.stream.start()

    def audio_callback(self, indata, frames, time, status):
        if status:
            print("Stream status:", status)
        self.data = indata[:, 0].copy()

    def update(self):
        self.curve.setData(self.x, self.data)

    def run(self):
        self.win.show()
        QtWidgets.QApplication.instance().exec()

if __name__ == "__main__":
    scope = Scope()
    scope.run()