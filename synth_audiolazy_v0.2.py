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


# -----------drum test---------------------#
#---------hihat----------#

def hihat_1(hp_f,env_t1,env_l1,env_t2,env_l2,env_t3):
    hihat_src = white_noise()
    hihat_filter = highpass(hp_f * Hz)
    v_env = line(env_t1 * s, 1,env_l1).append(line(env_t2 * s, env_l1,env_l2)).append(line(env_t3 * s, env_l2,0))
    return (hihat_filter(0.2 * hihat_src.copy())) * v_env.copy()

# hihat_closed_sound_1 = hihat_1(16000,0.1,0.2,0.01,0.2,0.1)
# hihat_open_sound_1 = hihat_1(16000,0.2,0.4,0.01,0.4,0.2)

hihat_closed_sound_2 = hihat_1(16000,0.01,0.9,0.03,0.2,0.01)
hihat_open_sound_2 = hihat_1(15000,0.02,0.6,0.13,0.5,0.01)

#--------bass drum------------#

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

bass_drum_sound_2 = bass_drum_1(27.5,400,55,0.02,
                0.4,0.01,0.5,
                0.3 ,0.001,0.5,
                0.2,0.01,0.05)

# bass_drum_sound_1 = bass_drum_1(110,200,55,0.02,
#                 0.2,0.01,0.25,
#                 0.5,0.001,0.3,
#                 0.1,0.01,0.12)


#--------snare drum----------#

def snare_1(tr_f_t,tr_f_h,tr_f_l,tr_d,
            wn_hp_f,
            wn_env_1_t,wn_env_1_l,
            wn_env_2_t,wn_env_2_l,
            wn_env_3_t,wn_env_3_l,
            tr_mix
            ):
    tr_f = line(tr_f_t * s, tr_f_h, tr_f_l).append(tr_f_l * ones())
    tr = oscillator('triangle', tr_f).sig.copy()

    wn = white_noise()
    wn_hp = highpass(wn_hp_f * Hz)
    wn_env = line(wn_env_1_t * s, wn_env_1_l, wn_env_2_l).append(line(wn_env_2_t * s, wn_env_2_l, wn_env_3_l)).append(line(wn_env_3_t * s, wn_env_3_l, 0))
    snare_ = Streamix()
    snare_.add(0, 0.3 * tr_mix * (tr.copy()) * fadeout(tr_d * s))
    snare_.add(0, 0.3 * (1-tr_mix) * wn_hp(wn.copy()) * wn_env.copy())
    return snare_

snare_sound_1 = snare_1(tr_f_t=0.01,tr_f_h=4000,tr_f_l=220,tr_d=0.02,
            wn_hp_f=10000,
            wn_env_1_t=0.03,wn_env_1_l=1,
            wn_env_2_t=0.08,wn_env_2_l=0.9,
            wn_env_3_t=0.03,wn_env_3_l=0.4,
            tr_mix=0.5
            )

snare_sound_2 = snare_1(tr_f_t=0.02,tr_f_h=4000,tr_f_l=220,tr_d=0.02,
            wn_hp_f=5000,
            wn_env_1_t=0.03,wn_env_1_l=0.7,
            wn_env_2_t=0.02,wn_env_2_l=1,
            wn_env_3_t=0.08,wn_env_3_l=0.3,
            tr_mix=0.8
            )
# -------- test beat -----------#

test_beat = beatmaker(480,40)
test_beat.create_cycle(bass_drum_sound_2,1,[0,4,5,10,14,15,20,21,30,35])
test_beat.create_cycle(hihat_closed_sound_2,0.6,[0,2,3,4,5,7,8,9,
                                               10,12,13,14,15,17,18,19,
                                               20,22,23,24,25,27,29,
                                               30,32,33,34,35,37,38,39
                                               ])
test_beat.create_cycle(hihat_open_sound_2,0.6,[1,11,21,28,31])

test_beat.create_cycle(snare_sound_1,1,[6,16,26,32,36])
test_beat.create_cycle(0.7*snare_sound_2,1,[7,12,13,27,33,37,38,39])
test_beat_track = test_beat.generate_op_signal('num_cycles',4)


# ------------- synth patches ----------------------#

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
# [0,3,5,6,8],notes=['D4']
def bass_bars(bass_patch,bpm,bar_l,beats,notes,env_h,bars):
    len_beat = 60/bpm

    bass_bars_ = Streamix()
    bass_bars_.add(0,zeros(bar_l*len_beat*s))
    for i,b in enumerate(beats):
        bass_bars_.add(b*len_beat*s,bass_patch(f_l=notes[notes[i]],env_h=env_h[i]))

    return bass_bars_

bass_patch_op = Streamix()
bass_patch_op.add(0*s,bass_bars(bass_patch_like,480,10,[0,3,5,6,8],['D2']*5,[0.15,0.1,0.03,0.1,0.1]).copy())
bass_patch_op.add(1.25*s,bass_bars(bass_patch_like,480,10,[0,3,5,6,8],['D2']*5,[0.15,0.1,0.03,0.1,0.1]).copy())
bass_patch_op.add(1.25*s,bass_bars(bass_patch_like,480,10,[0,3,5,6,8],['D2']*5,[0.15,0.1,0.03,0.1,0.1]).copy())
bass_patch_op.add(1.25*s,bass_bars(bass_patch_like,480,10,[0,3,5,6,8],['D2']*5,[0.15,0.1,0.03,0.1,0.1]).copy())

bass_patch_op.add(0*s,bass_bars(bass_patch_like,480,10,[0,3,5,6,8],['C2']*5,[0.15,0.1,0.03,0.1,0.1]).copy())
bass_patch_op.add(1.25*s,bass_bars(bass_patch_like,480,10,[0,3,5,6,8],['C2']*5,[0.15,0.1,0.03,0.1,0.1]).copy())
bass_patch_op.add(1.25*s,bass_bars(bass_patch_like,480,10,[0,3,5,6,8],['C2']*5,[0.15,0.1,0.03,0.1,0.1]).copy())
bass_patch_op.add(1.25*s,bass_bars(bass_patch_like,480,10,[0,3,5,6,8],['C2']*5,[0.15,0.1,0.03,0.1,0.1]).copy())


rhodes_like = partial(patch_1_notes,f_a=30,f_f_f=300,f_f_a=110,f_f_l=220,
                      env_at=0.01,env_h=0.1,env_d1=0.2,env_d1_l=0.6,env_d2=3,env_f=1.5,env_a=0.1)

test_synth_op = Streamix()

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


# organ_like = partial(string_attempt_1,hrm=[-2,-1,1,2,3],hrm_a=2,
#                      env_at_1=0.7,env_at_1_l=0.4,env_at_2=0.6,
#                      env_d_1=0.8,env_d_1_l=0.3,env_d_2=1,
#                      voices=1,voices_l=0.3,voices_det=2,voices_rng=rng)

organ_like = partial(string_attempt_1,hrm=[-1,1,2,3,4],hrm_a=2.3,
                     env_at_1=0.7,env_at_1_l=0.4,env_at_2=0.6,
                     env_d_1=0.8,env_d_1_l=0.3,env_d_2=1,
                     voices=1,voices_l=0.4,voices_det=2,voices_rng=rng)


organ_sound_op = Streamix()
organ_sound_op_h_1 = 3.6
organ_sound_op.add(0,organ_like(f_l=notes['D4'],env_h=organ_sound_op_h_1,lp_f=notes['D4']).copy())
organ_sound_op.add(0,organ_like(f_l=notes['F4'],env_h=organ_sound_op_h_1,lp_f=notes['F4']).copy())
organ_sound_op.add(0*s,organ_like(f_l=notes['A4'],env_h=organ_sound_op_h_1,lp_f=notes['A4']).copy())

organ_sound_op.add(5*s,organ_like(f_l=notes['C4'],env_h=organ_sound_op_h_1,lp_f=notes['C4']).copy())
organ_sound_op.add(0,organ_like(f_l=notes['E4'],env_h=organ_sound_op_h_1,lp_f=notes['E4']).copy())
organ_sound_op.add(0,organ_like(f_l=notes['G4'],env_h=organ_sound_op_h_1,lp_f=notes['G4']).copy())
organ_sound_op.add(0,organ_like(f_l=notes['B4'],env_h=organ_sound_op_h_1,lp_f=notes['B4']).copy())

organ_sound_op.add(5*s,organ_like(f_l=notes['F4'],env_h=organ_sound_op_h_1,lp_f=notes['F4']).copy())
organ_sound_op.add(0*s,organ_like(f_l=notes['A4'],env_h=organ_sound_op_h_1,lp_f=notes['A4']).copy())
organ_sound_op.add(0*s,organ_like(f_l=notes['C5'],env_h=organ_sound_op_h_1,lp_f=notes['C5']).copy())
organ_sound_op.add(0*s,organ_like(f_l=notes['E5'],env_h=organ_sound_op_h_1,lp_f=notes['E5']).copy())

organ_sound_op.add(5*s,organ_like(f_l=notes['E4'],env_h=organ_sound_op_h_1,lp_f=notes['E4']).copy())
organ_sound_op.add(0*s,organ_like(f_l=notes['G4'],env_h=organ_sound_op_h_1,lp_f=notes['G4']).copy())
organ_sound_op.add(0*s,organ_like(f_l=notes['B4'],env_h=organ_sound_op_h_1,lp_f=notes['B4']).copy())
organ_sound_op.add(0*s,organ_like(f_l=notes['D5'],env_h=organ_sound_op_h_1,lp_f=notes['D5']).copy())

organ_sound_op_2 = Streamix()
organ_sound_op_2_h_1 = 3.6
organ_sound_op_2.add(0,organ_like(f_l=notes['A4'],env_h=organ_sound_op_2_h_1,lp_f=notes['A4']).copy())
organ_sound_op_2.add(0,organ_like(f_l=notes['C5'],env_h=organ_sound_op_2_h_1,lp_f=notes['C5']).copy())
organ_sound_op_2.add(0*s,organ_like(f_l=notes['E5'],env_h=organ_sound_op_2_h_1,lp_f=notes['E5']).copy())
organ_sound_op_2.add(0*s,organ_like(f_l=notes['G5'],env_h=organ_sound_op_2_h_1,lp_f=notes['G5']).copy())

organ_sound_op_2.add(5*s,organ_like(f_l=notes['C4'],env_h=organ_sound_op_2_h_1,lp_f=notes['C4']).copy())
organ_sound_op_2.add(0,organ_like(f_l=notes['E4'],env_h=organ_sound_op_2_h_1,lp_f=notes['E4']).copy())
organ_sound_op_2.add(0,organ_like(f_l=notes['G4'],env_h=organ_sound_op_2_h_1,lp_f=notes['G4']).copy())
organ_sound_op_2.add(0,organ_like(f_l=notes['B4'],env_h=organ_sound_op_2_h_1,lp_f=notes['B4']).copy())

organ_sound_op_2.add(5*s,organ_like(f_l=notes['G4'],env_h=organ_sound_op_2_h_1,lp_f=notes['G4']).copy())
organ_sound_op_2.add(0*s,organ_like(f_l=notes['B4'],env_h=organ_sound_op_2_h_1,lp_f=notes['B4']).copy())
organ_sound_op_2.add(0*s,organ_like(f_l=notes['D5'],env_h=organ_sound_op_2_h_1,lp_f=notes['D5']).copy())

organ_sound_op_2.add(5*s,organ_like(f_l=notes['A3'],env_h=2*organ_sound_op_2_h_1,lp_f=notes['A3']).copy())
organ_sound_op_2.add(0,organ_like(f_l=notes['C4'],env_h=2*organ_sound_op_2_h_1,lp_f=notes['C4']).copy())
organ_sound_op_2.add(0*s,organ_like(f_l=notes['E4'],env_h=2*organ_sound_op_2_h_1,lp_f=notes['E4']).copy())



#---final track0---#
test_final_track_2 = Streamix()
test_final_track_2.add(0,0.14*organ_sound_op.copy())
test_final_track_2.add(20*s,0.14*organ_sound_op.copy())
test_final_track_2.add(0,test_beat_track.copy())
test_final_track_2.add(20*s,0.14*organ_sound_op_2.copy())
test_final_track_2.add(0,test_beat_track.copy())

#-------------

def write_to_file(str_mix,t,path):
    print('Writing file')
    str_mix.add(0,zeros())
    str_mix_array = np.array(list(str_mix.peek(t*s)),dtype='float32')
    str_mix_array = (0.99/max(str_mix_array))*str_mix_array
    print(max(str_mix_array))
    wv.write(filename=path,rate=rate,data=str_mix_array)
    return str_mix_array

organ_chords_beat_array = write_to_file(test_final_track_2,70,'../output/organ_chords_beat.wav')
print("Done!")
with AudioIO(True) as player:
    player.play(organ_chords_beat_array, rate=rate)
