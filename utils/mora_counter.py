# coding: utf-8

import argparse
import re
from os.path import splitext
from pdb import set_trace

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inf")
    parser.add_argument("outf")
    parser.add_argument("--target_columun", type=int, default=1)
    return parser.parse_args()

def moraCounter(string):
    cnt = 0
    for s in string:
        if s=="ー":
            cnt += 1
    for e in (" ", "\u3000", "。", "．", "、", "，", "ー"):
        string = string.replace(e, "")
    exclude = list("ぁぃぅぇぉっゃゅょァィゥェォッャュョ")
    for e in exclude:
        string = string.replace(e, "")
    # 1. To check the input is Hiragana or Katakana
    if re.compile(r'^[あ-ん]+$').fullmatch(string):
        pass
    elif re.compile(r'[\u30A1-\u30F4]+').fullmatch(string):
        pass
    else:
        assert False, "入力は[ひらがな]か[カタカナ]だけである必要があります"
    return len(string) + cnt

def main():
    assert splitext(args.inf)[1].lower() in (".csv", ".tsv"), "This script supports {tsv, csv} file only for $inf."
    assert splitext(args.inf)[1].lower() in (".csv", ".tsv"), "This script supports {tsv, csv} file only for $outf."
    with open(args.inf) as fd:
        Data = fd.read().strip().split("\n")
    in_delimiter = "," if splitext(args.inf)[1].lower()==".csv" else "\t"
    out_delimiter = "," if splitext(args.outf)[1].lower()==".csv" else "\t"
    out = []
    for data in Data:
        data = data.split(in_delimiter)
        mora = moraCounter(data[args.target_columun])
        out.append(out_delimiter.join(data) + out_delimiter + str(mora))
    with open(args.outf, "w") as fd:
        fd.write("\n".join(out)+"\n")
    print("%s is saved." %args.outf)
    return 0

if __name__=="__main__":
    args = get_args()
    main()
