from audiolazy import *
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import scipy.io.wavfile as wv
rate = 44100
s, Hz = sHz(rate)
ms = 1e-3 * s
notes =  {'C0':16.35,'C#0':17.32,'D0':18.35,'D#0':19.45,'E0':20.60,'F0':21.83,'F#0':23.12,'G0':24.50,
          'G#0':25.96,'A0':27.50,'A#0':29.14,'B0':30.87,'C1':32.70,'C#1':34.65,'D1':36.71,'D#1':38.89,
          'E1':41.20,'F1':43.65,'F#1':46.25,'G1':49.00,'G#1':51.91,'A1':55.00,'A#1':58.27,'B1':61.74,
          'C2':65.41,'C#2':69.30,'D2':73.42,'D#2':77.78,'E2':82.41,'F2':87.31,'F#2':92.50,'G2':98.00,
          'G#2':103.83,'A2':110.00,'A#2':116.54,'B2':123.47,'C3':130.81,'C#3':138.59,'D3':146.83,'D#3':155.56,
          'E3':164.81,'F3':174.61,'F#3':185.00,'G3':196.00,'G#3':207.65,'A3':220.00,'A#3':233.08,'B3':246.94,
          'C4':261.63,'C#4':277.18,'D4':293.66,'D#4':311.13,'E4':329.63,'F4':349.23,'F#4':369.99,'G4':392.00,
          'G#4':415.30,'A4':440.0,'A#4':466.16,'B4':493.88,'C5':523.25,'C#5':554.37,'D5': 587.33,'D#5': 622.25,
          'E5':659.26,'F5':698.46,'F#5':739.99,'G5':783.99,'G#5': 830.61,'A5': 880.0,'A#5':932.33,'B5':987.77,
          'C6':1046.50,'C#6':1108.73,'D6':1174.66,'D#6':1244.51,'E6':1318.51,'F6':1396.91,'F#6':1479.98,'G6':1567.98,
          'G#6':1661.22,'A6':1760.00,'A#6':1864.66,'B6':1975.53,'C7':2093.00,'C#7':2217.46,'D7':2349.32,'D#7':2489.02,
          'E7':2637.02,'F7':2793.83,'F#7':2959.96,'G7':3135.96,'G#7':3322.44,'A7':3520.00,'A#7':3729.31,'B7':3951.07,
          'C8':4186.01,'C#8':4434.92,'D8':4698.63,'D#8':4978.03,'E8':5274.04,'F8':5587.65,'F#8':5919.91,'G8':6271.93,
           'G#8':6644.88,'A8':7040.00,'A#8':7458.62,'B8':7902.13
          }

rng = np.random.default_rng(seed=1234)
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

    def __init__(self, type, f, amp=1.0, level=0.0):
        """
        :param type: can be one of ['sin','square','saw','reverse_saw','triangle']
        :param f: frequency of oscillator in Hertz
        """
        if type not in ['sin', 'square', 'saw', 'reverse_saw', 'triangle']:
            raise ValueError("type must be one of ['sin','square','saw','reverse_saw','triangle']")

        if type == 'sin':
            self.sig = level + amp * sinusoid(f * Hz)
        elif type == 'square':
            self.sig = level + amp * (floor(sinusoid(f * Hz)) + ceil(sinusoid(f * Hz)))
        elif type == 'saw':
            cycle_samples = ceil(s / f)
            one_wave = line(cycle_samples, 0, 1, finish=True)
            self.sig = level + amp * cycle(one_wave)
        elif type == 'reverse_saw':
            cycle_samples = ceil(s / f)
            one_wave = line(cycle_samples, 1, 0, finish=True)
            self.sig = level + amp * cycle(one_wave)
        elif type == 'triangle':
            int_filt = z / (z - 1)
            if isinstance(f,Stream):
                scale_factor = (1 / s) * (int_filt(f)/int_filt(ones()))
            else:
                scale_factor = f / s
            self.sig = level - scale_factor/2 + scale_factor * amp * int_filt(floor(sinusoid(f * Hz)) + ceil(sinusoid(f * Hz)))

class track:
    def __init__(self):
        self.track_sig = [(zeros(), 0)]
        self.track_vol = ones()

    def add_to_track(self, sig, wt):
        """
        :param sig: signal to add to track (~ instrument)
        :param wt: weight in the mix (between (0,1])
        :return:
        """
        if wt <= 0 or wt > 1:
            raise ValueError("Value of wt must be (0,1]")
        self.track_sig.append((sig, wt))

    def generate_op_signal(self, dur):
        """

        :param dur: duration of track in seconds. This will be overriden by duration of volume envelope.
        :return: mixed track
        """
        wt_scale = 1 / (sum([sig[1] for sig in self.track_sig]))
        op_sig = Streamix()
        for sig in self.track_sig:
            op_sig.add(0,wt_scale * sig[1] * sig[0].copy())
        op_sig = op_sig.copy() * self.track_vol.copy()
        return op_sig

class beatmaker:
    def __init__(self, bpm, cycle_length_beats):
        self.bpm = bpm
        self.beat_length = (60 / bpm) * s
        self.cycle_length_beats = cycle_length_beats
        self.cycle_length = cycle_length_beats*self.beat_length
        self.track_sig = [(zeros(), 0)]

    def create_cycle(self, sample, wt, beat_program):
        """
        :param sample: signal of sample to add to track (~ instrument)
        :param wt: weight in the mix (between (0,1])
        :return:
        """
        if wt <= 0 or wt > 1:
            raise ValueError("Value of wt must be (0,1]")
        if max(beat_program) >= self.cycle_length_beats:
            raise ValueError("Invalid beat programming. Beat value cannot be greater than cycle length")
        beat_program = list(dict.fromkeys(beat_program))

        cycle_sig = Streamix()

        for i,v in enumerate(beat_program):
            # cycle_sig = cycle_sig.tee() + zeros(i*self.beat_length).append(_beat.tee()).append(zeros((self.cycle_length_beats-1-i)*self.beat_length))
            if i==0:
                cycle_sig.add(v*self.beat_length,sample.copy())
            else:
                cycle_sig.add((v - beat_program[i-1]) * self.beat_length, sample.copy())
        self.track_sig.append((cycle_sig, wt))

    def generate_op_signal(self, type, dur):
        """

        :param dur: duration of track in seconds. This will be overriden by duration of volume envelope.
        :return: mixed track
        """
        wt_scale = 1 / (sum([sig[1] for sig in self.track_sig]))
        op_sig_cycle = Streamix()
        for sig in self.track_sig:
            op_sig_cycle.add(0, wt_scale * sig[1] * sig[0].copy())

        if type == 'time':
            num_cycles = ceil(dur*s/self.cycle_length)
        elif type == 'num_cycles':
            num_cycles = dur

        op_sig = Streamix()
        op_sig.add(0, op_sig_cycle.copy())
        for i in range(num_cycles-1):
            op_sig.add(self.cycle_length,op_sig_cycle.copy())

        return op_sig


# ----------basic track test-----------#

track_dur = 30 * s
filter_dur = 20 * s
fadeout_dur = 3 * s

# basic oscillator track with vol envelope
test_track = track()
test_track.track_vol = 0.5 * ones(track_dur).append(fadeout(fadeout_dur))
test_osc = oscillator(type='square', f=110)
test_track.add_to_track(test_osc.sig, 1)
test_op = test_track.generate_op_signal(track_dur + fadeout_dur)

# filtering using bandpass filter
lfo_f = line(filter_dur, 4, 110).append(
    110 * ones())  # increasing linearly from 4 to 110 over track_dur and then constant
lfo_1 = oscillator(type='sin', f=lfo_f, amp=400, level=200)

freq1 = lfo_1.sig  # oscillating as per lfo_1
freq2 = line(0.2 * s, 8000, 0).append(zeros())  # decreasing linearly from 8000 to 0 in 0.2 s and then constant at 0
bw = line(filter_dur, 1000, 100).append(100 * ones())  # band width

filt = resonator((freq1 + freq2) * Hz, bw * Hz)  # bandpass filter

test_op_filt = filt(test_op)


# -----------drum test---------------------#
#---------hihat----------#
hihat_test1 = white_noise()
# hihat_test = oscillator(type='square',f=15000).sig + white_noise()
# hihat_test_filter_f = line(0.02 * s, 11000, 11000).append(11000 * ones())
hihat_test_filter_f = 16000
# hihat_test_filter = resonator(hihat_test_filter_f * Hz, 5000 * Hz)
hihat_test_filter = highpass(hihat_test_filter_f * Hz)
closed_hihat_vol_env = line(0.1 * s, 1,0.2).append(line(0.1 * s, 0.2,0))
open_hihat_vol_env = line(0.2 * s, 1,0.4).append(line(0.2 * s, 0.4,0))
# hihat_sound_op = hihat_test1.copy() * fadeout(0.1 * s)
hihat_closed_sound_op = (hihat_test_filter(0.2*hihat_test1.copy())) * closed_hihat_vol_env.copy()
hihat_open_sound_op = (hihat_test_filter(0.2*hihat_test1.copy())) * open_hihat_vol_env.copy()
# hihat_sound_op = Streamix()
# hihat_sound_op.add(0,fadein(0.04 * s))
# hihat_sound_op.add(0,hihat_test2.peek(0.1*s))
# hihat_sound_op.add(0.12*s,fadeout(0.04 * s))

def hihat_1(hp_f,env_t1,env_l1,env_t2,env_l2,env_t3):
    hihat_src = white_noise()
    hihat_filter = highpass(hp_f * Hz)
    v_env = line(env_t1 * s, 1,env_l1).append(line(env_t2 * s, env_l1,env_l2)).append(line(env_t3 * s, env_l2,0))
    return (hihat_filter(0.2 * hihat_src.copy())) * v_env.copy()

# hihat_closed_sound_1 = hihat_1(16000,0.1,0.2,0.01,0.2,0.1)
# hihat_open_sound_1 = hihat_1(16000,0.2,0.4,0.01,0.4,0.2)

hihat_closed_sound_1 = hihat_1(16000,0.01,0.9,0.05,0.2,0.01)
hihat_open_sound_1 = hihat_1(15000,0.02,0.6,0.13,0.5,0.01)



#--------bass drum------------#
bass_test1 = oscillator('sin',110).sig.copy()
# bass_test2 = oscillator('saw',110).sig.copy()
bass_test2 = white_noise()
bass_test_3_f = line(0.02 * s, 200, 55 ).append(55 * ones())
bass_test3 = oscillator('sin',bass_test_3_f).sig.copy()

# bass_test_filter_f = line(0.01 * s, 500 * Hz, 50 * Hz).append(50 * Hz * ones())
# bass_test_filter = resonator(bass_test_filter_f , 100 * Hz)
bass_test_filter = lowpass(bass_test_3_f * Hz)
bass_sound_op = Streamix()
bass_sound_op.add(0,0.2*(fadein(0.01 * s).append(ones()))*(bass_test1.copy()) * fadeout(0.25 * s))
bass_sound_op.add(0,0.1*(fadein(0.01 * s).append(ones()))*(bass_test_filter(bass_test2.copy())) * fadeout(0.12 * s))
bass_sound_op.add(0,0.5*(bass_test3.copy()) * fadeout (0.3  * s))

def bass_drum_1(sin_1_f,sin_2_f1,sin_2_f2,sin_2_f1f2_t,
                sin_1_l,sin_1_at,sin_1_d,
                sin_2_l,sin_2_at,sin_2_d,
                wn_l,wn_at,wn_d):
    bass_sin1 = oscillator('sin', sin_1_f).sig.copy()
    bass_wn = white_noise()
    bass_sin2_f = line(sin_2_f1f2_t * s, sin_2_f1, sin_2_f2).append(sin_2_f2 * ones())
    bass_sin2 = oscillator('sin', bass_sin2_f).sig.copy()

    bass_lp_filter = lowpass(bass_sin2_f * Hz)
    bass_drum_op = Streamix()
    bass_drum_op.add(0, sin_1_l * (fadein(sin_1_at * s).append(ones())) * (bass_sin1.copy()) * fadeout(sin_1_d * s))
    bass_drum_op.add(0, wn_l * (fadein(wn_at * s).append(ones())) * (bass_lp_filter(bass_wn.copy())) * fadeout(wn_d * s))
    bass_drum_op.add(0, sin_2_l * (fadein(sin_2_at * s).append(ones())) * (bass_sin2.copy()) * fadeout(sin_2_d * s))
    return bass_drum_op

bass_drum_sound_2 = bass_drum_1(27.5,400,55,0.01,
                0.4,0.01,0.15,
                0.3 ,0.001,0.15,
                0.03,0.01,0.05)

# bass_drum_sound_1 = bass_drum_1(110,200,55,0.02,
#                 0.2,0.01,0.25,
#                 0.5,0.001,0.3,
#                 0.1,0.01,0.12)


#--------snare drum----------#
snare_test1_f = line(0.01 * s,4000,220).append(220 * ones())
# snare_test1_f = 220
snare_test1 =  oscillator('triangle',snare_test1_f).sig.copy()

# print(max(list(snare_test1.peek(2*s))))
snare_test2 =  white_noise()
snare_test2_filter_f = 10000
snare_test2_filter = highpass(snare_test2_filter_f * Hz)
snare_test_2_vol_env = line(0.03 * s, 1,0.9).append(line(0.08 * s, 0.9,0.4)).append(line(0.03 * s, 0.4,0))
snare_sound_op = Streamix()
snare_sound_op.add(0,0.3 * (snare_test1.copy()) * fadeout(0.02* s))
snare_sound_op.add(0,0.3 * snare_test2_filter(snare_test2.copy()) * snare_test_2_vol_env.copy())

# -------- test beat -----------#
# test_beat = beatmaker(480,16)
# test_beat.create_cycle(hihat_closed_sound_1,0.6,[0,4,6,8,9,12,14])
# test_beat.create_cycle(hihat_open_sound_1,0.6,[2,10])
# test_beat.create_cycle(bass_drum_sound_1,1,[0,4,8,12,15])
# test_beat.create_cycle(snare_sound_op,0.6,[4,12])
# test_beat_track = test_beat.generate_op_signal('num_cycles',8)

test_beat_2 = beatmaker(320,16)
# test_beat_2.create_cycle(hihat_closed_sound_1,0.6,[0,1,2,4,5,6,7,8,9,10,11,12,13,14])
test_beat_2.create_cycle(hihat_closed_sound_1,0.4,[0,2,4,6,8,10,12,14])
# test_beat_2.create_cycle(hihat_open_sound_1,0.6,[3,15])
test_beat_2.create_cycle(bass_drum_sound_2,1,[0,3,6,10,14])
test_beat_2.create_cycle(snare_sound_op,0.8,[4,12])
test_beat_2_track = test_beat_2.generate_op_signal('num_cycles',4)

test_beat_3 = beatmaker(320,16)
test_beat_3.create_cycle(hihat_closed_sound_1,0.4,[0,1,2,4,5,6,7,8,9,10,11,12,13,14])
test_beat_3.create_cycle(hihat_open_sound_1,0.4,[3,15])
test_beat_3.create_cycle(bass_drum_sound_2,1,[0,3,6,9,10,14])
test_beat_3.create_cycle(snare_sound_op,0.8,[4,12])
test_beat_3_track = test_beat_3.generate_op_signal('num_cycles',4)

test_beat_4 = beatmaker(320,16)
test_beat_4.create_cycle(hihat_closed_sound_1,0.4,[0,1,2,4,5,6,7,8,9,10,11,12,13,14])
test_beat_4.create_cycle(hihat_open_sound_1,0.4,[3,15])
test_beat_4.create_cycle(bass_drum_sound_2,1,[0,3,6,9,10,14])
test_beat_4.create_cycle(snare_sound_op,0.8,[4,12])
test_beat_4_track = test_beat_4.generate_op_signal('num_cycles',3)

test_beat_4_r = beatmaker(320,16)
test_beat_4_r.create_cycle(hihat_closed_sound_1,0.4,[0,1,2,4,5,6,7,8,9,10,11,12,13,14])
test_beat_4_r.create_cycle(hihat_open_sound_1,0.4,[3,15])
test_beat_4_r.create_cycle(bass_drum_sound_2,1,[0,3,6,9,10,11,13])
test_beat_4_r.create_cycle(snare_sound_op,0.8,[4,12,14,15])
test_beat_4_r_track = test_beat_4_r.generate_op_signal('num_cycles',1)


#---adding to track----#
# test_final_track = Streamix()

# test_final_track.add(0,0.12*test_op_filt.copy())
# test_final_track.add(22*s,test_beat_track.copy())

# test = ones(10)
# test_filt = z/(z-1)

# (z/(z-1)).plot().show()
# print(list(test.copy()))
# print(list(test_filt(test).copy()))

# test_osc = oscillator('triangle',440).sig.copy()

# plt.plot(test_hihat_cycle.copy().take(2*s))
# plt.show()

# ------------- synth patches ----------------------#

# test_synth_A_f_intr = (440 + 20*(fadein(0.01 * s).append(fadeout(0.02 * s))))
# test_synth_A_f = test_synth_A_f_intr.copy().append(oscillator('square',200,30,440).sig.copy())
# test_synth_A_f_m = oscillator('sin',10,30,440).sig.copy()
# test_synth_A_f = oscillator('square',200,30,test_synth_A_f_m).sig.copy()

# test_synth_f_f = (50 + 100*(fadein(0.8 * s).append(fadeout(0.8 * s)).append(zeros())))
# test_synth_f_f = 220
# test_synth_f_f = oscillator('sin', 300, 110, 220).sig.copy()  #
# test_synth_f_a = 30 # good till 50-60

def patch_1_notes(f_l,f_a,
                  f_f_f,f_f_a,f_f_l,
                  env_at,env_h,env_d1,env_d1_l,env_d2,env_f,env_a
                  ):
    f_f = oscillator('sin', f_f_f, f_f_a, f_f_l).sig.copy()  #
    f = oscillator('sin', f_f.copy(), f_a, f_l).sig.copy()
    # test_synth_env = fadein(0.01 * s).append(ones(0.1 * s)).append(line(0.2 * s, 1, 0.6)).append(0.6 * fadeout(3 * s) * oscillator('sin', 1.5, 0.1, 1).sig.copy())
    env = (fadein(env_at * s).append(ones(env_h * s)).append(line(env_d1 * s, 1, env_d1_l)).append(env_d1_l * fadeout(env_d2 * s))
           ) * oscillator('sin', env_f, env_a, 1).sig.copy()
    return env.copy()*oscillator('sin', f.copy()).sig.copy()

def patch_2_notes(f_l,f_a_r,f_f,
                  lp_f,
                  env_h,
                  env_at_1, env_d1_1, env_d1_l_1, env_d2_1,
                  env_at_2, env_d1_2, env_d1_l_2, env_d2_2,
                  sm):
    adj_1 = 0.01 + max(env_at_1+env_d1_1+env_d2_1,env_at_2+env_d1_2+env_d2_2) - (env_at_1+env_d1_1+env_d2_1)
    synth_1 = oscillator('square',f_l).sig.copy()
    lp_1 = lowpass(lp_f * Hz)
    env_1 = (fadein(env_at_1 * s).append(ones(env_h * s)).append(line(env_d1_1 * s, 1, env_d1_l_1)).append(
        env_d1_l_1 * fadeout(env_d2_1 * s))).append(zeros(adj_1*s))

    adj_2 = 0.01 * s + max(env_at_1 + env_d1_1 + env_d2_1, env_at_2 + env_d1_2 + env_d2_2) - (env_at_2 + env_d1_2 + env_d2_2)
    synth_2_f = oscillator('sin',f_f,f_a_r*f_l,f_l).sig.copy()
    synth_2 = oscillator('sin',synth_2_f).sig.copy()
    env_2 = (fadein(env_at_2 * s).append(ones(env_h * s)).append(line(env_d1_2 * s, 1, env_d1_l_2)).append(
        env_d1_l_2 * fadeout(env_d2_2 * s))).append(zeros(adj_2*s))
    return  (sm*env_2*synth_2.copy() + (1-sm)*env_1*lp_1(synth_1.copy()))

bass_patch_like = partial(patch_2_notes,
                       f_a_r=0,f_f=100,
                       lp_f=50,
                       env_at_1=0.005,env_d1_1=0.1,env_d1_l_1=0.4,env_d2_1=0.3,
                       env_at_2=0.05,env_d1_2=0.4,env_d1_l_2=0.6,env_d2_2=0.3,
                       sm=0.55)

bass_patch_op = Streamix()
bass_patch_op.add(0*s,0.2*bass_patch_like(f_l=notes['D2'],env_h=0.2).copy())
bass_patch_op.add(1.125*s,0.2*bass_patch_like(f_l=notes['D2'],env_h=0.1).copy())
bass_patch_op.add(0.75*s,0.2*bass_patch_like(f_l=notes['D2'],env_h=0.1).copy())

bass_patch_op.add(1.125*s,0.2*bass_patch_like(f_l=notes['B1'],env_h=0.2).copy())
bass_patch_op.add(1.125*s,0.2*bass_patch_like(f_l=notes['B1'],env_h=0.1).copy())
bass_patch_op.add(0.75*s,0.2*bass_patch_like(f_l=notes['B1'],env_h=0.1).copy())

bass_patch_op.add(1.125*s,0.2*bass_patch_like(f_l=notes['A1'],env_h=0.2).copy())
bass_patch_op.add(1.125*s,0.2*bass_patch_like(f_l=notes['A1'],env_h=0.1).copy())
bass_patch_op.add(0.75*s,0.2*bass_patch_like(f_l=notes['A1'],env_h=0.1).copy())

bass_patch_op.add(1.125*s,0.2*bass_patch_like(f_l=notes['A1'],env_h=0.2).copy())
bass_patch_op.add(1.125*s,0.2*bass_patch_like(f_l=notes['A1'],env_h=0.1).copy())
bass_patch_op.add(0.75*s,0.2*bass_patch_like(f_l=notes['A1'],env_h=0.1).copy())

# test_synth_A_f_m = 440
# test_synth_A_f = oscillator('sin',test_synth_f_f.copy(),test_synth_f_a,test_synth_A_f_m).sig.copy()
# test_synth_A = oscillator('sin',test_synth_A_f.copy()).sig.copy()
#
# test_synth_C_f_m = 523.25
# test_synth_C_f = oscillator('sin',test_synth_f_f.copy(),test_synth_f_a,test_synth_C_f_m).sig.copy()
# test_synth_C = oscillator('sin',test_synth_C_f.copy()).sig.copy()
#
# test_synth_E_f_m = 659.26
# test_synth_E_f = oscillator('sin',test_synth_f_f.copy(),test_synth_f_a,test_synth_E_f_m).sig.copy()
# test_synth_E = oscillator('sin',test_synth_E_f.copy()).sig.copy()

# test_synth_4_f_m = 554.37
# test_synth_4_f = oscillator('sin',test_synth_f_f.copy(),test_synth_f_a,test_synth_4_f_m).sig.copy()
# test_synth_4 = oscillator('sin',test_synth_4_f.copy()).sig.copy()

# notes_init = 'A4,A#4,B4,C5,C#5,D5,D#5,E5,F5,F#5,G5,G#5,A5'.split(',')
# freqs = np.round(440. * 2**(np.arange(0, len(notes_init)) / 12.),2)
# notes = dict(zip(notes_init, freqs))

# print(notes)
# print(notes['C5'])

rhodes_like = partial(patch_1_notes,f_a=30,f_f_f=300,f_f_a=110,f_f_l=220,
                      env_at=0.01,env_h=0.1,env_d1=0.2,env_d1_l=0.6,env_d2=3,env_f=1.5,env_a=0.1)


# rhodes_like = partial(patch_1_notes,f_a=0,f_f_f=30,f_f_a=2,f_f_l=30,
#                       env_at=0.01,env_h=0.1,env_d1=0.2,env_d1_l=0.6,env_d2=3,env_f=1.5,env_a=0.1)

# test_synth_C_f = test_synth_A_f_intr.copy().append(440*ones())
# test_synth_C = oscillator('sin',test_synth_C_f).sig.copy()
# test_synth_env = fadein(0.05 * s).append(ones(2 * s)).append(fadeout(1 * s))
# test_synth_env = fadein(0.01 * s).append(ones(0.1*s)).append(line(0.5 * s,1,0.4)).append(0.4*fadeout(2 * s))
# test_synth_env = fadein(0.01 * s).append(ones(0.1*s)).append(line(0.2*s,1,0.6)).append(0.6*fadeout(3*s)*oscillator('sin',1.5,0.1,1).sig.copy())
# test_synth_filter_f = 600 + 100*(fadein(0.05 * s).append(line(0.5 * s,1,0.3)).append(0.3*fadeout(1 * s)).append(zeros()))
# test_synth_filter_f =400
# test_synth_filter_bw = 50
# test_synth_filter = resonator(test_synth_filter_f,test_synth_filter_bw)
test_synth_op = Streamix()
test_synth_op.add(0*s,0.2*rhodes_like(f_l=notes['D4']).copy())
test_synth_op.add(0.08*s,0.2*rhodes_like(f_l=notes['F4']).copy())
test_synth_op.add(0.12*s,0.2*rhodes_like(f_l=notes['A4']).copy())
test_synth_op.add(0.15*s,0.2*rhodes_like(f_l=notes['C5']).copy())

test_synth_op.add((3-0.08-0.12-0.15)*s,0.2*rhodes_like(f_l=notes['B4']).copy())
test_synth_op.add(0.07*s,0.2*rhodes_like(f_l=notes['D5']).copy())
test_synth_op.add(0.13*s,0.2*rhodes_like(f_l=notes['F5']).copy())

test_synth_op.add((3-0.07-0.13)*s,0.2*rhodes_like(f_l=notes['A4']).copy())
test_synth_op.add(0.09*s,0.2*rhodes_like(f_l=notes['C5']).copy())
test_synth_op.add(0.12*s,0.2*rhodes_like(f_l=notes['E5']).copy())

#----------- strings ------------------#

def string_attempt_1(f_l,hrm,hrm_a,
                     env_at_1,env_at_1_l,env_at_2,
                     env_h,
                     env_d_1,env_d_1_l,env_d_2,
                     voices,voices_l,voices_det,voices_rng,
                     lp_f
                     ):

    env = line(env_at_1 * s, 0, env_at_1_l).append(line(env_at_2 * s, env_at_1_l,1)).append(ones(env_h * s)).append(
        line(env_d_1 * s, 1, env_d_1_l)).append(line(env_d_2*s,env_d_1_l,0))

    str_op = Streamix()
    hrm = [0] + hrm
    amps = [hrm_a ** (-1 * abs(h)) for h in hrm]
    for i,h in enumerate(hrm):
        a = amps[i]/(sum(amps)*(1+voices*voices_l))
        m = 2 ** h
        str_op.add(0, a * oscillator('saw', m * f_l).sig.copy())
        for v in range(voices):
            det_delta  = voices_rng.uniform(-voices_det,voices_det)
            str_op.add(0,voices_l * a * oscillator('sin', m*(f_l) + det_delta).sig.copy())

    lp = lowpass(lp_f * Hz)
    return 0.8*env *lp(str_op.copy())

organ_sound_1 = string_attempt_1(440,[-1,1,2,3],2,
                     1,0.4,0.6,
                     3,
                     0.5,0.3,0.8,
                     1,0.3,2,rng,
                     400
                     )

print(max(list(organ_sound_1.peek(5*s))))

organ_like = partial(string_attempt_1,hrm=[1,2,3],hrm_a=2,
                     env_at_1=0.7,env_at_1_l=0.4,env_at_2=0.6,
                     env_d_1=0.5,env_d_1_l=0.3,env_d_2=0.8,
                     voices=1,voices_l=0.3,voices_det=2,voices_rng=rng)

organ_like_2 = partial(string_attempt_1,hrm=[-2,-1,1,2,3],hrm_a=2,
                     env_at_1=0.7,env_at_1_l=0.4,env_at_2=0.6,
                     env_d_1=0.8,env_d_1_l=0.3,env_d_2=1,
                     voices=1,voices_l=0.3,voices_det=2,voices_rng=rng)

organ_sound_op = Streamix()

organ_sound_op.add(0,organ_like(f_l=notes['D4'],env_h=2,lp_f=notes['D4']).copy())
organ_sound_op.add(0,organ_like(f_l=notes['F4'],env_h=2,lp_f=notes['F4']).copy())
organ_sound_op.add(0,organ_like(f_l=notes['A4'],env_h=2,lp_f=notes['A4']).copy())
organ_sound_op.add(0,organ_like(f_l=notes['C5'],env_h=2,lp_f=notes['C5']).copy())

organ_sound_op.add(2.7*s,organ_like(f_l=notes['B4'],env_h=2,lp_f=notes['B4']).copy())
organ_sound_op.add(0,organ_like(f_l=notes['D5'],env_h=2,lp_f=notes['D5']).copy())
organ_sound_op.add(0,organ_like(f_l=notes['F5'],env_h=2,lp_f=notes['F5']).copy())

organ_sound_op.add(2.7*s,organ_like(f_l=notes['A4'],env_h=4,lp_f=notes['A4']).copy())
organ_sound_op.add(0*s,organ_like(f_l=notes['C5'],env_h=4,lp_f=notes['C5']).copy())
organ_sound_op.add(0*s,organ_like(f_l=notes['E5'],env_h=4,lp_f=notes['E5']).copy())

organ_sound_op_2 = Streamix()

organ_sound_op_2.add(0,organ_like_2(f_l=notes['D4'],env_h=2,lp_f=notes['D4']).copy())
organ_sound_op_2.add(0,organ_like_2(f_l=notes['F4'],env_h=2,lp_f=notes['F4']).copy())
organ_sound_op_2.add(0,organ_like_2(f_l=notes['A4'],env_h=2,lp_f=notes['A4']).copy())
organ_sound_op_2.add(0,organ_like_2(f_l=notes['C5'],env_h=2,lp_f=notes['C5']).copy())

organ_sound_op_2.add(2.7*s,organ_like_2(f_l=notes['B4'],env_h=2,lp_f=notes['B4']).copy())
organ_sound_op_2.add(0,organ_like_2(f_l=notes['D5'],env_h=2,lp_f=notes['D5']).copy())
organ_sound_op_2.add(0,organ_like_2(f_l=notes['F5'],env_h=2,lp_f=notes['F5']).copy())

organ_sound_op_2.add(2.7*s,organ_like_2(f_l=notes['A4'],env_h=4,lp_f=notes['A4']).copy())
organ_sound_op_2.add(0*s,organ_like_2(f_l=notes['C5'],env_h=4,lp_f=notes['C5']).copy())
organ_sound_op_2.add(0*s,organ_like_2(f_l=notes['E5'],env_h=4,lp_f=notes['E5']).copy())


test_final_track_2 = Streamix()
# bar 1
test_final_track_2.add(0.5*s,0.2*test_synth_op.copy())
test_final_track_2.add(0,0.9*test_beat_2_track.copy())
test_final_track_2.add(0,bass_patch_op.copy())
# bar 2
test_final_track_2.add(12*s,0.9*test_beat_2_track.copy())
test_final_track_2.add(0,0.2*test_synth_op.copy())
test_final_track_2.add(0,bass_patch_op.copy())
# bar 3
test_final_track_2.add(12*s,0.9*test_beat_3_track.copy())
test_final_track_2.add(0,0.2*test_synth_op.copy())
test_final_track_2.add(0,bass_patch_op.copy())
# bar 4
test_final_track_2.add(12*s,0.9*test_beat_4_track.copy())
test_final_track_2.add(0,0.2*test_synth_op.copy())
test_final_track_2.add(0,bass_patch_op.copy())
test_final_track_2.add(9*s,0.9*test_beat_4_r_track.copy())
# bar 5
test_final_track_2.add(3*s,0.9*test_beat_3_track.copy())
test_final_track_2.add(0,0.2*test_synth_op.copy())
test_final_track_2.add(0,bass_patch_op.copy())
test_final_track_2.add(0,0.12*organ_sound_op.copy())
# bar 6
test_final_track_2.add(12*s,0.9*test_beat_3_track.copy())
test_final_track_2.add(0,0.2*test_synth_op.copy())
test_final_track_2.add(0,bass_patch_op.copy())
test_final_track_2.add(0,0.13*organ_sound_op.copy())
# bar 7
test_final_track_2.add(12*s,0.9*test_beat_3_track.copy())
test_final_track_2.add(0,0.2*test_synth_op.copy())
test_final_track_2.add(0,bass_patch_op.copy())
test_final_track_2.add(0,0.14*organ_sound_op.copy())
# bar 8
test_final_track_2.add(12*s,0.22*test_synth_op.copy())
test_final_track_2.add(0,0.18*organ_sound_op_2.copy())

# test_synth_op = Streamix()
# test_synth_op.add(0,0.2*test_synth_A.copy())
# test_synth_op.add(0.09*s,0.2*test_synth_C.copy())
# test_synth_op.add(0.13*s,0.2*test_synth_E.copy())

# test_synth_op = test_synth_filter(test_synth_C.copy()).peek(4*s)
# test_synth_op = test_synth_C.copy().peek(4*s)

# print(max(list(test_synth_op.copy())))

def write_to_file(str_mix,t,path):
    print('Writing file')
    str_mix.add(0,zeros())
    str_mix_array = np.array(list(str_mix.peek(t*s)),dtype='float32')
    str_mix_array = (0.99/max(str_mix_array))*str_mix_array
    print(max(str_mix_array))
    wv.write(filename=path,rate=rate,data=str_mix_array)
    return str_mix_array

# organ_sound_op_array = write_to_file(organ_sound_op,15,'../output/organ_chords.wav')
test_final_track_2_array = write_to_file(test_final_track_2,100,'../output/lowfi_hh_2.wav')
print("Done!")
with AudioIO(True) as player:
    player.play(test_final_track_2_array, rate=rate)
