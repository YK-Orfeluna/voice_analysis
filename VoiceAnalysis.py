# coding: utf-8

import argparse
from os import makedirs
from os.path import basename, splitext
import numpy as np
from scipy import stats
from scipy.io import wavfile
from tqdm import tqdm
import matplotlib.pyplot as plt
try:
    import webrtcvad
except ImportError:
    exit("Import Error: You have to install 'py-webrtcvad'.\nTo run 'pip install webrtcvad'.")
try:
    import pysptk
except ImportError:
    exit("Import Error: You have to install 'pysptk'.\nTo run 'pip install pysptk'.")
try:
    import pyreaper
except ImportError:
    exit("Import Error: You have to install 'pyreaper'.\nTo run 'pip install pyreaper'.")

from pdb import set_trace

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inlist")
    parser.add_argument("outd")
    ### VAD options
    parser.add_argument("--vad_mode", type=int, choices=(0, 1, 2, 3), default=1, \
        help="To see py-webrtcvad's document; default is '1'.")
    parser.add_argument("--vad_frame_duration", type=int, choices=(10, 20, 30), default=10, \
        help="To see py-webrtcvad's document; default is '10' [millsec.]")
    ### F0 options
    parser.add_argument("--f0_range", type=float, default=(40.0, 500.0), nargs=2, \
        help="To see pyreaper's document; default is '(40.0, 500.0)'.")
    parser.add_argument("--f0_period", type=int, default=5, \
        help="To see pyreaper's dotument; default is '5' [millsec.]")
    ### power options
    parser.add_argument("--power_period", type=int, default=5, \
        help="Frame period for computing voice power; default is '5' [millsec.]")
    parser.add_argument("--power_type", default="rms", choices=("rms", "db"))
    ### demo
    parser.add_argument("--demo", action="store_true", default=False, \
        help="If activate, demo will be displayed.")
    parser.add_argument("--demo_index", type=int, default=-1)
    return parser.parse_args()

def VAD(data, rate):
    vad = webrtcvad.Vad()
    vad.set_mode(args.vad_mode)
    i = 0
    length = int(rate / 1000 * args.vad_frame_duration)
    out = np.array([], dtype=data.dtype)
    rslt = []
    while True:
        _rslt = [vad.is_speech(data[i: i+length].tobytes(), rate)] * int(args.vad_frame_duration / 1000 * rate)
        rslt.extend(_rslt)
        if rslt[-1]:
            out = np.append(out, data[i: i+length]) 
        i += length
        if i==data.shape[0]:
            break
        elif i+length>data.shape[0]:
            _rslt = [vad.is_speech(data[-length: ].tobytes(), rate)] * int((i-length) / 1000 * rate)
            rslt.append(vad.is_speech(data[-length: ].tobytes(), rate))
            if rslt[-1]:
                out = np.append(out, data[i: ])
            break
    return out, rslt
    # [duration]秒ごとのvad結果を1フレームごとに作り替える

def F0(data, rate):
    pm_times, pm, f0_times, f0, corr = \
        pyreaper.reaper(data, rate, minf0=min(args.f0_range), maxf0=max(args.f0_range), frame_period=args.f0_period/1000)
    return f0

def Power(data, rate):
    rslt = []
    i = 0
    length = int(rate * args.power_period / 1000)
    while i<data.shape[0]:
        rslt.append(np.sqrt(np.mean(data[i: i+length].astype(np.float32)**2)))
        i += length
    rslt = np.array(rslt, dtype=np.float32)
    if args.power_type=="db":
        rslt = np.log10(rslt) * 20
    return rslt

def compute_stats(data):
    mean = np.mean(data)    # 平均値
    var = np.var(data)    # 分散
    median = np.median(data)    # 中央値
    mode, mode_cnt = stats.mode(data)    # 最頻値とその出現回数
    vmin = min(data)    # 最小値
    vmax = max(data)    # 最大値
    q75, q25 = np.percentile(data, [75, 25])    # 第3, 第1四分位点
    iqr = q75 - q25    # 四分位範囲
    skew = stats.skew(data)    # 尤度
    kurtosis = stats.kurtosis(data)    # 尖度
    return mean, var, median, mode[0], mode_cnt[0]/data.shape[0], vmin, vmax, q75, q25, iqr, skew, kurtosis

def demo(inf=None):
    if inf is None:
        inf = pysptk.util.example_audio_file()
    rate, data = wavfile.read(inf)
    assert rate in (8000, 16000, 32000, 48000), "py-webrtcvad corresponds to {8000, 16000, 32000, 48000} Hz. Your input is %s Hz." %rate
    vad_wav, vad_flag = VAD(data, rate)
    wavfile.write("%s/original.wav" %args.outd, rate, data)
    wavfile.write("%s/vad.wav" %args.outd, rate, vad_wav)
    f0 = F0(vad_wav, rate)
    power = Power(vad_wav, rate)
    fig = plt.figure(figsize=(12, 10))
    # left side
    plt.subplot(321)
    plt.plot(data, color="blue")
    plt.title("original PCM (sampling rate: %s Hz)" %rate)
    plt.ylim(max(np.abs(data))*-1, max(np.abs(data)))
    plt.subplot(323)
    plt.plot(vad_flag, color="red")
    plt.title("Results of VAD")
    plt.yticks((0.0, 1.0), ("not voice", "voice"))
    # right side
    plt.subplot(322)
    plt.plot(vad_wav, color="cyan")
    plt.title("PCM after VAD")
    plt.ylim(max(np.abs(data))*-1, max(np.abs(data)))
    plt.subplot(324)
    plt.plot(f0, color="green")
    plt.title("F0")
    plt.ylabel("[Hz]")
    plt.subplot(326)
    plt.plot(power, color="orange")
    plt.title("power")
    plt.ylabel("[RMS]" if args.power_type=="rms" else "[dB]")
    plt.tight_layout()
    plt.show()
    outf = "%s/demo.png" %args.outd
    fig.savefig(outf, dpi=300)
    print("%s is saved." %outf)
    return 0

header="filename\tmean\tvariance\tmedian\tmode\tmode_ratio\tmin\tmax\tq75\tq25\tIQR\tskew\tkurtosis\n"
def analysis(infs):
    f0_out = header
    power_out = header
    for inf in tqdm(infs):
        rate, data = wavfile.read(inf)
        assert rate in (8000, 16000, 32000, 48000), "py-webrtcvad corresponds to {8000, 16000, 32000, 48000} Hz. Your input is %s Hz." %rate
        vad_wav, _ = VAD(data, rate)
        f0 = F0(data, rate)
        f0 = f0[f0>-1]
        f0_stats = compute_stats(f0)
        f0_out += inf+"\t"+"\t".join(map(str, f0_stats))+"\n"
        power = Power(data, rate)
        power_stats = compute_stats(power)
        power_out  += inf+"\t"+"\t".join(map(str, power_stats))+"\n"
    for name, out in zip(("f0", "power"), (f0_out, power_out)):
        outf = "%s/%s.tsv" %(args.outd, name)
        with open(outf, "w") as fd:
            fd.write(out)
        print("%s is saved." %args.outd)
    return 0

def main():
    if args.demo:
        demo()
        return 0
    with open(args.inlist) as fd:
        infs = sorted(fd.read().strip().split("\n"))
    if args.demo_index>=0:
        demo(infs[args.demo_index])
        return 0
    analysis(infs)

if __name__=="__main__":
    args = get_args()
    makedirs(args.outd, exist_ok=True)
    main()

"""
1. VADを実施して、非音声区間を除外する
2. 時間窓で{F0, powerを計算}
3. {F0, power}の統計量を求める
4. 発話区間長、{F0, power}の統計量を出力

マルチプロセスはとりあえず実装しない
VADやF0のパラメタは全部argparseで指定できるようにする
wavファイルを前提として、サンプリングレートはsoxなどで編集するものとする

パラメタの確認用として、「VADのデモ; 全波形と音声 or notが見えるような図、VAD後の音声を出力する」を作っておく
"""