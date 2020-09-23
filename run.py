# coding: utf-8

import argparse
from os import makedirs
from os.path import basename, splitext
import numpy as np
from scipy import stats
from scipy.io import wavfile
import webrtcvad
import pysptk
import pyreaper

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inlist")
    parser.add_argument("outd")
    ### VAD options
    parser.add_argument("--vad_mode", type=int, choices=(0, 1, 2, 3), default=1, \
        help="To see py-webrtcvad's document; default=1.")
    parser.add_argument("--vad_frame_dutarion", type=int, choices=(10, 20, 30), default=10, \
        help="To see py-webrtcvad's document; default=10.")
    ### F0 options
    parser.add_argument("--f0_range", type=float, default=(40.0, 500.0), nargs=2, \
        help="To see pyreaper's document; default=(40.0, 500.0).")
    parser.add_argument("--f0_period", type=float, default=0.005, \
        help="To see pyreaper's dotument; default=0.005.")
    ### power options
    parser.add_argument("--power_period", type=float, default=0.005, \
        help="Frame period for computing voice power; default=0.005(sec.).")
    return parser.parse_args()

def VAD(data, rate):
    vad = webrtcvad.Vad()
    vad.set_mode(args.vad_mode)
    i = 0
    length = int(rate / 1000 * args.vad_frame_dutarion)
    out = np.array([], dtype=data.dtype)
    rslt = []
    while True:
        rslt.append(vad.is_speech(data[i: i+length].tobytes()))
        if rslt[-1]:
            out = np.append(out, data[i: i+length]) 
        i += length
        if i==data.shape[0]:
            break
        elif i>data.shape[0]:
            rslt.append(vad.is_speech(data[-length: ].tobytes()))
            if rslt[-1]:
                out = np.append(out, data[i: ])
            break
    return out, rslt
    # [duration]秒ごとのvad結果を1フレームごとに作り替える

def F0(data, rate):
    pm_times, pm, f0_times, f0, corr = pyreaper.reaper(date, rate, minf0=min(args.f0_range), maxf0=max(args.f0_range), frame_period=args.f0_period)
    return f0[f0>-1]

def power(data, rate):
    rslt = []
    i = 0
    length = int(rate * args.power_period)
    while i<data.shape[0]:
        rslt.append(np.sqrt(np.mean(np.sum(data[i: i+length].astype(np.float32)**2))))
        i += length
    return np.array(rslt, dtype=np.float32)

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
    return mean, var, median, mode, mode_cnt, vmin, vmax, q75, q25, iqr, skew, kurtosis

def main():
    with open(args.inlist) as fd:
        infs = sorted(fd.read().strip().split("\n"))
    for inf in tqdm(infs):
        rate, data = wavfile.read(inf)
        assert rate in (8000, 16000, 32000, 48000), "py-webrtcvad corresponds to 8k, 16k, 32k or 48k Hz. Your input is %s Hz." %rate

# if __name__=="__main__":
#     args = get_args()
#     makedirs(args.outd, exist_ok=True)

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