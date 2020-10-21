# -*- coding:utf-8 -*-
# -*- coding: utf-8 -*-
import sys

reload(sys)
sys.setdefaultencoding("utf-8")

import re
import numpy as np
import pandas as pd
from keras.layers import (
        Dense,
        Embedding,
        LSTM,
        TimeDistributed,
        Input,
        Bidirectional,
 )
    from keras.models import Model


def clean(s):
    return (
        s.replace(u'"  ', "").replace(u'" ', "").replace(u"’ ", "").replace(u"‘ ", "")
    )


def get_xy(s):
    s = re.findall("(.)/(.)", s)
    if s:
        s = np.array(s)
        return list(s[:, 0]), list(s[:, 1])


def main():
    maxlen = 32
    lines = open("msr_training.utf8").read()  # .decode('gbk')

    lines = clean(lines)
    lines = lines.split("\n")

    file_ob = open("sentence.txt", "w")
    newlines = []
    for line in lines:
        temlines = re.split(u"[,.!?/:]", line)
        for line in temlines:
            line = line.replace("\n", "").strip()
            if len(line):
                newlines.append(line)
                file_ob.write(line + "\n")
    file_ob.close()

    print len(newlines)
    data = []  # 生成训练样本
    label = []
    return

    for i in s:
        x = get_xy(i)
        if x:
            data.append(x[0])
            label.append(x[1])

    print len(data)
    d = pd.DataFrame(index=range(len(data)))
    d["data"] = data
    d["label"] = label
    print data
    print label
    return
    d = d[d["data"].apply(len) <= maxlen]
    d.index = range(len(d))
    tag = pd.Series({"s": 0, "b": 1, "m": 2, "e": 3, "x": 4})

    chars = [] 
    for i in data:
        chars.extend(i)

    chars = pd.Series(chars).value_counts()
    chars[:] = range(1, len(chars) + 1)

    # 生成适合模型输入的格式
    from keras.utils import np_utils

    d["x"] = d["data"].apply(
        lambda x: np.array(list(chars[x]) + [0] * (maxlen - len(x)))
    )
    d["y"] = d["label"].apply(
        lambda x: np.array(
            map(lambda y: np_utils.to_categorical(y, 5), tag[x].reshape((-1, 1)))
            + [np.array([[0, 0, 0, 0, 1]])] * (maxlen - len(x))
        )
    )

    # 设计模型
    word_size = 128
    maxlen = 32


    sequence = Input(shape=(maxlen,), dtype="int32")
    embedded = Embedding(
        len(chars) + 1, word_size, input_length=maxlen, mask_zero=True
    )(sequence)
    blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode="sum")(embedded)
    output = TimeDistributed(Dense(5, activation="softmax"))(blstm)
    model = Model(input=sequence, output=output)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    batch_size = 1024
    history = model.fit(
        np.array(list(d["x"])),
        np.array(list(d["y"])).reshape((-1, maxlen, 5)),
        batch_size=batch_size,
        nb_epoch=50,
    )


if __name__ == "__main__":
    main()
