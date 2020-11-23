# coding: utf-8

import argparse
from os import makedirs
from os.path import basename, splitext, abspath
import subprocess as sp

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inlist", \
        help="List of input files as text style.")
    parser.add_argument("outd", \
        help="Converted files are saved in $outd.")
    parser.add_argument("--in_sampling_rate", type=int, \
        help="If you will convert without wav file, to choose sampling rate of input files.")
    parser.add_argument("--in_bit_rate", type=int, default=16, \
        help="If you will convert without wav file, to choose bit rate of input files; default is '16' bit.")
    parser.add_argument("--in_channel", type=int, choices=(1, 2), \
        help="If you will convert without wav file, to choose # of channels of input files.")
    parser.add_argument("--out_sampling_rate", type=int, choices=(8000, 16000, 32000, 48000), default=16000, \
        help="To choose sampling rate of output files; default is '16000' Hz.")
    return parser.parse_args()

def main():
    with open(args.inlist) as fd:
        inlist = fd.read().strip().split("\n")
    makedirs(args.outd, exist_ok=True)
    outlist = []
    print("Now converting...")
    for inf in inlist:
        outf = "%s/%s.wav" %(args.outd, splitext(basename(inf))[0])
        if splitext(inf)[1].lower()==".wav":
            cmd = ["sox", \
                inf, \
                "-r", args.out_sampling_rate, "-b", "16", "-e", "signed-integer", "-c", "1", outf\
            ]
        else:
            cmd = ["sox", \
                "-t", "raw", "-r", args.in_sampling_rate, "-b", "-c", args.in_channel, inf, \
                "-t", "wav", "-r", args.out_sampling_rate, "-b", "16", "-e", "signed-integer", "-c", "1", outf\
            ]
        sp.run(list(map(str, cmd)))
        outlist.append(abspath(outf))
    with open("%s/wav.list", "w") as fd:
        fd.write("\n".join(sorted(outlist))+"\n")
    print("All input files has been converted, and %s/wav.list is saved." %args.outd)
    return 0

if __name__=="__main__":
    args = get_args()
    main()
            