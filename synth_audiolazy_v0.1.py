import scipy as sc
from audiolazy import *

rate = 44100
s, Hz = sHz(rate)
ms = 1e-3 * s

note_f = 110
note_a = 0.3
sig1 = note_a*floor(sinusoid(note_f * Hz)) + note_a*ceil(sinusoid(note_f * Hz))
sig2 = white_noise()
mix_wt = 0.99
sig =  mix_wt * sig1 + (1-mix_wt) * sig2

env = ones(25*s).append(fadeout(2*s))
# print(sig.take(100))
dur_bw_f = 20 * s # Some few seconds of audio
dur_f2 = 0.2 * s # Some few seconds of audio
# freq = line(dur, 800, 200).append(200*ones())
# lfo_f = 4
lfo_f = line(dur_bw_f,4,110).append(110*ones())
freq1 = 200 + 400*(1+sinusoid(lfo_f * Hz))
freq2 = line(dur_f2,8000,0).append(zeros())
bw = line(dur_bw_f, 1000, 100).append(100*ones())
# print(freq.take(500))
# freq = line(dur, 8000, 50).append(50*fadeout(5*s)) # A lazy iterable range
# bw = line(dur, 240, 100).append(100*fadeout(5*s))

filt = resonator((freq1+freq2) * Hz, bw * Hz) # A simple bandpass filter
# filt = lowpass(freq * Hz)
with AudioIO(True) as player:
  player.play(filt(sig*env), rate=rate)

# rate = 44100 # Sampling rate, in samples/second
# s, Hz = sHz(rate) # Seconds and hertz
# ms = 1e-3 * s
# note1 = karplus_strong(440 * Hz) # Pluck "digitar" synth
# note2 = zeros(300 * ms).append(karplus_strong(880 * Hz))
# note3 = zeros(600 * ms).append(karplus_strong(550 * Hz))
# notes = (1.1*note1 + 0.7*note2 + note3) * .5
# notes2 = notes
# for i in range(5):
#     notes2 = notes2.append(zeros(600 * ms))
#     notes2 = notes2.append(notes)
# sound = notes.take(int(4 * s))  # 2 seconds of a Karplus-Strong note
# with AudioIO(True) as player:  # True means "wait for all sounds to stop"
#     player.play(sound, rate=rate)