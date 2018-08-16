ChirpChirp is a script to transmit data over audio with chirp modulation.


## Requirements

ChirpChirp requires Python, [Numpy][], [PortAudio][], and [PyAudio][].

To install these on a Debian GNU/Linux system:

    $ sudo apt-get install python python-numpy python-pyaudio


## Usage

First choose a chirp, which is determined by its bounding frequencies
and period (duration).  For example, frequencies of 2 to 12 kHz,
and a period of one eighth of a second, are a good choice.

Then start a receiving process in one terminal:

    $ python -u chirpchirp.py rx 2000 12000 0.125

and a transmitting process in another terminal, with the same parameters:

    $ echo Hello | python -u chirpchirp.py tx 2000 12000 0.125


[Python]: <https://www.python.org/>
[NumPy]: <http://www.numpy.org/>
[PyAudio]: <https://people.csail.mit.edu/hubert/pyaudio/>
[PortAudio]: <http://www.portaudio.com/>
