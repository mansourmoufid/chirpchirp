#!/usr/bin/env python -u

'''chirpchirp.py - transmit data over audio with chirp modulation

Usage:
    python -u chirpchirp.py (tx | rx) <fmin> <fmax> <period> [<volume>]
    python chirpchirp.py -h

Options:
    -h  Print this help.
'''


from __future__ import print_function

import functools
import numpy
import os
import pyaudio
try:
    import queue
except ImportError:
    import Queue as queue
import sys
import termios


__author__ = 'Mansour Moufid'
__email__ = 'mansourmoufid@gmail.com'
__copyright__ = 'Copyright 2018, Mansour Moufid'
__license__ = 'ISC'
__version__ = '0.1'
__status__ = 'Development'


PCM_DTYPE = numpy.int16

PCM_MAX = 2.0 ** (16 - 1) - 1.0


def setnoncanonical(fd):
    assert os.isatty(fd)
    attr = termios.tcgetattr(fd)
    lflags = attr[3]
    lflags &= ~termios.ICANON
    attr[3] = lflags
    termios.tcsetattr(fd, termios.TCSANOW, attr)


def readbytes(f, tty=None):
    while True:
        char = f.read(1)
        if char == '':
            break
        byte = ord(char)
        if tty and byte == 4:
            break
        yield byte


def tobits(bytes):
    for byte in bytes:
        for i in range(8):
            yield (byte >> i) & 1


def tobyte(bits):
    return sum(2 ** i * bit for i, bit in enumerate(bits))


def modxcor(x, y):
    m = x.size
    n = y.size
    x = numpy.copy(x)
    y = numpy.copy(y[::-1])
    x.resize(m + n)
    y.resize(m + n)
    X = numpy.fft.rfft(x)
    Y = numpy.fft.rfft(y)
    Z = X * Y
    z = numpy.fft.irfft(Z, n=(m + n))
    return z[(n / 2):-(n / 2)]


linspace = functools.partial(
    numpy.linspace,
    dtype=numpy.float32,
    endpoint=False,
)


def chirp(bandwidth, period, samples):
    t = linspace(0.0, period, num=samples)
    fmin, fmax = bandwidth
    k = (fmax - fmin) / period
    f = t * k / 2 + fmin
    return numpy.sin(2.0 * numpy.pi * f * t)


def mod(zero, one, bit):
    data = one if bit == 1 else zero
    pcm = PCM_DTYPE(data * A * PCM_MAX)
    frames = pcm.tostring()
    return frames


def dem(zero, one, bits, frames, nframes=None, timing=None, status=None):
    pcm = numpy.fromstring(frames, dtype=PCM_DTYPE)
    data = numpy.float32(pcm) / (PCM_MAX + 1.0)
    xc0 = modxcor(data, zero)
    xc1 = modxcor(data, one)
    i0 = numpy.argmax(xc0 * xc0)
    i1 = numpy.argmax(xc1 * xc1)
    if abs(xc0[i0]) > 10.0 * numpy.std(xc0):
        bits.put(0)
    if abs(xc1[i1]) > 10.0 * numpy.std(xc1):
        bits.put(1)
    return (None, pyaudio.paContinue)


if __name__ == '__main__':

    if len(sys.argv) == 2 and sys.argv[1] == '-h':
        print(__doc__)
        sys.exit(0)
    try:
        assert sys.argv[1] in ['tx', 'rx']
        tx = sys.argv[1] == 'tx'
        fmin = int(sys.argv[2])
        fmax = int(sys.argv[3])
        period = float(sys.argv[4])
        try:
            A = float(sys.argv[5])
        except:
            A = 1.0
    except:
        print(__doc__)
        sys.exit(os.EX_USAGE)

    audio = pyaudio.PyAudio()
    if tx:
        info = audio.get_default_output_device_info()
    else:
        info = audio.get_default_input_device_info()
    fs = int(info['defaultSampleRate'])
    samples = int(fs * period)
    audio.open = functools.partial(
        audio.open,
        channels=1,
        format=pyaudio.paInt16,
        rate=fs,
    )

    one = chirp((fmin, fmax), period, samples)
    zero = one[::-1]

    if tx:
        stream = audio.open(
            frames_per_buffer=one.size,
            output=True,
        )
        if sys.stdin.isatty():
            setnoncanonical(sys.stdin.fileno())
            tty = os.ctermid()
        else:
            tty = None
        for bit in tobits(readbytes(sys.stdin, tty=tty)):
            frames = mod(zero, one, bit)
            stream.write(frames)
        if tty:
            with open(tty, 'w') as f:
                f.write('\n')
    else:
        q = queue.Queue()
        decode = functools.partial(dem, zero, one, q)
        stream = audio.open(
            frames_per_buffer=one.size,
            input=True,
            stream_callback=decode,
        )
        stream.start_stream()
        bits = []
        while stream.is_active():
            try:
                bit = q.get_nowait()
            except queue.Empty:
                continue
            bits.append(bit)
            if len(bits) == 8:
                byte = tobyte(bits)
                if sys.stdout.isatty():
                    byte &= 127
                sys.stdout.write(chr(byte))
                bits = []

    stream.stop_stream()
    stream.close()
    audio.terminate()
