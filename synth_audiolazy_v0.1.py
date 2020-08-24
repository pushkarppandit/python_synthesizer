from audiolazy import *

rate = 44100
s, Hz = sHz(rate)
ms = 1e-3 * s

# note_f = 110
# note_a = 0.3
# sig1 = note_a*floor(sinusoid(note_f * Hz)) + note_a*ceil(sinusoid(note_f * Hz))
# sig2 = white_noise()
# mix_wt = 0.99
# sig =  mix_wt * sig1 + (1-mix_wt) * sig2
#
# env = ones(25*s).append(fadeout(2*s))
# # print(sig.take(100))
# dur_bw_f = 20 * s # Some few seconds of audio
# dur_f2 = 0.2 * s # Some few seconds of audio
# # freq = line(dur, 800, 200).append(200*ones())
# # lfo_f = 4
# lfo_f = line(dur_bw_f,4,110).append(110*ones())
# freq1 = 200 + 400*(1+sinusoid(lfo_f * Hz))
# freq2 = line(dur_f2,8000,0).append(zeros())
# bw = line(dur_bw_f, 1000, 100).append(100*ones())
# print(freq.take(500))
# freq = line(dur, 8000, 50).append(50*fadeout(5*s)) # A lazy iterable range
# bw = line(dur, 240, 100).append(100*fadeout(5*s))

# filt = resonator((freq1+freq2) * Hz, bw * Hz) # A simple bandpass filter
# # filt = lowpass(freq * Hz)


class oscillator:
  """
  Generates an oscillation whose range is [-1,1]
  """
  def __init__(self,type,f,amp=1,level=0):
    """
    :param type: can be one of ['sin','square','saw','reverse_saw','triangle']
    :param f: frequency of oscillator in Hertz
    """
    if type not in ['sin','square','saw','reverse_saw','triangle']:
      raise ValueError("type must be one of ['sin','square','saw','reverse_saw','triangle']")

    if type=='sin':
      self.sig = level + amp*sinusoid(f * Hz)
    elif type=='square':
      self.sig = level + amp*(floor(sinusoid(f * Hz)) + ceil(sinusoid(f * Hz)))
    elif type=='saw':
      cycle_samples = ceil(s/f)
      one_wave = line(cycle_samples,0,1,finish=True)
      self.sig = level + amp*Stream(*list(one_wave))
    elif type=='reverse_saw':
      cycle_samples = ceil(s/f)
      one_wave = line(cycle_samples,1,0,finish=True)
      self.sig = level + amp*Stream(*list(one_wave))
    elif type=='triangle':
      half_cycle_samples = ceil(s/(2*f))
      one_wave = line(half_cycle_samples,0,1,finish=True).append(line(half_cycle_samples,1,0,finish=True))
      self.sig = level + amp*Stream(*list(one_wave))

class track:
  def __init__(self):
    track.track_sig = [(zeros(),0)]
    track.track_vol = ones()

  def add_to_track(self,sig,wt):
    """
    :param sig: signal to add to track (~ instrument)
    :param wt: weight in the mix (between (0,1])
    :return:
    """
    if wt <= 0 or wt > 1:
      raise ValueError("Value of wt must be (0,1]")
    self.track_sig.append((sig,wt))

  def generate_op_signal(self,dur):
    """

    :param dur: duration of track in seconds. This will be overriden by duration of volume envelope.
    :return: mixed track
    """
    wt_scale = 1/(sum([sig[1] for sig in self.track_sig]))
    op_sig = zeros(dur*s)
    for sig in self.track_sig:
      op_sig = op_sig + wt_scale*sig[1]*sig[0]
    op_sig = op_sig*self.track_vol
    return op_sig


#----------basic track test-----------#
track_dur = 20 * s
fadeout_dur = 3 * s

# basic oscillator track with vol envelope
test_track = track()
test_track.track_vol = 0.5*ones(track_dur).append(fadeout(fadeout_dur))
test_osc = oscillator(type='square',f=110)
test_track.add_to_track(test_osc.sig,1)
test_op = test_track.generate_op_signal(track_dur+fadeout_dur)

# filtering using bandpass filter
lfo_f = line(track_dur,4,110).append(110*ones())  # increasing linearly from 4 to 110 over track_dur and then constant
lfo_1 = oscillator(type='sin',f=lfo_f,amp=400,level=200)

freq1 = lfo_1.sig  # oscillating as per lfo_1
freq2 = line(0.2*s,8000,0).append(zeros())  # decreasing linearly from 8000 to 0 in 0.2 s and then constant at 0
bw = line(track_dur, 1000, 100).append(100*ones())  # band width

filt = resonator((freq1+freq2) * Hz, bw * Hz)  # bandpass filter

test_op_filt = filt(test_op)

#-----------hihat test---------------------#
class beatmaker:
  def __init__(self,bpm,cycle_length):
    self.bpm = bpm
    self.beat_length = (60/bpm) * s
    self.cycle_length = cycle_length


hihat_test = white_noise()
# hihat_test = oscillator(type='square',f=15000).sig + white_noise()
hihat_test_filter_f = line(0.05*s,12000,9000).append(9000*ones())
# hihat_test_filter_f = 16000
hihat_test_filter = resonator(hihat_test_filter_f * Hz, 200 * Hz)
# hihat_test_filter = lowpass(hihat_test_filter_f * Hz)
hihat_sound_op = (hihat_test_filter(hihat_test))*fadeout(0.2*s)

with AudioIO(True) as player:
  player.play(hihat_sound_op, rate=rate)