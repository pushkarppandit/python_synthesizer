from audiolazy import *
import matplotlib.pyplot as plt

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

    def __init__(self, type, f, amp=1, level=0):
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
            if isinstance(f,int):
                scale_factor = f/s
            elif isinstance(f,Stream):
                scale_factor = (1 / s) * (int_filt(f)/int_filt(ones()))
            self.sig = level + scale_factor * amp * int_filt(floor(sinusoid(f * Hz)) + ceil(sinusoid(f * Hz)))

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

#--------snare drum----------#
snare_test1_f = line(0.05 * s,4000,220).append(220 * ones())
# snare_test1_f = 220
snare_test1 =  oscillator('triangle',snare_test1_f).sig.copy()

print(max(list(snare_test1.peek(2*s))))
snare_test2 =  white_noise()
snare_test2_filter_f = 10000
snare_test2_filter = highpass(snare_test2_filter_f * Hz)
snare_test_2_vol_env = line(0.08 * s, 1,0.3).append(line(0.1 * s, 0.3,0.1)).append(line(0.04 * s, 0.1,0))
snare_sound_op = Streamix()
snare_sound_op.add(0,0.3 * (snare_test1.copy()) * fadeout(0.1* s))
snare_sound_op.add(0,0.3 * snare_test2_filter(snare_test2.copy()) * snare_test_2_vol_env.copy())


test_beat = beatmaker(480,16)
test_beat.create_cycle(hihat_closed_sound_op,0.6,[0,4,6,8,9,12,14])
test_beat.create_cycle(hihat_open_sound_op,0.6,[2,10])
test_beat.create_cycle(bass_sound_op,1,[0,4,8,12,15])
test_beat.create_cycle(snare_sound_op,0.6,[4,12])


test_beat_track = test_beat.generate_op_signal('num_cycles',8)

test_final_track = Streamix()

test_final_track.add(0,0.12*test_op_filt.copy())
test_final_track.add(22*s,test_beat_track.copy())

# test = ones(10)
# test_filt = z/(z-1)

# (z/(z-1)).plot().show()
# print(list(test.copy()))
# print(list(test_filt(test).copy()))

# test_osc = oscillator('triangle',440).sig.copy()

# plt.plot(test_hihat_cycle.copy().take(2*s))
# plt.show()
print("Done!")
with AudioIO(True) as player:
    player.play(test_final_track, rate=rate)
