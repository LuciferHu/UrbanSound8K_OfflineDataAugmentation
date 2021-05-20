# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import os
import struct
import random
import soundfile as sf
import librosa
import make_dataframe
import audio_mix

"""
本文件针对类别，对数据进行增强
"""
fg_wav_dir = "/home/richardhu/Documents/PycharmProjects/ub_leaking_with_augmentation/data/audio"  # 前景音路径
bg_wav_dir = "/home/richardhu/Downloads/DCASE2020/train/16bit_audio/"  # 背景音路径
DCASE2020_4scene = make_dataframe.DCASE2020_4scene_df("/home/richardhu/Downloads/DCASE2020/train/meta.csv")
ub_leaking_nl = make_dataframe.ub_leaking_nl_df("./data/metadata/ub_leaking_nl.csv")
scenes = ['park', 'public_square', 'street_pedestrian', 'street_traffic']


def car_horn_aug(csv_dir):
    """
    用于对car_horn类的数据增强，这393个数据需要分别与DCASE四个scene的数据进行mix操作
    :param csv_dir: 获取ub_leaking_nl.csv路径: "./data/metadata/ub_leaking_nl.csv"
    :return:
    """
    global ub_leaking_nl
    car_horn_sample = make_dataframe.car_horn_sample_df(csv_dir)
    # print("car_horn_sample: ", car_horn_sample)
    for scenes_items in scenes:
        # items: park, public_square, street_pedestrian, street_traffic
        bg = DCASE2020_4scene.loc[DCASE2020_4scene["scene_label"] == scenes_items]  # 挑选了一个类别的背景音
        for index, items in car_horn_sample.iterrows():
            # index 范围：[0, 329]
            # items 内容：‘slice_file_name’, 'fold', 'classID', 'class'
            fg_wav = os.path.join(fg_wav_dir, 'fold' + str(items['fold']), items['slice_file_name'])  # 前景音文件路径
            bg = bg.sample(n=1)  # 注意bg是一个dataframe
            bg_wav = os.path.join(bg_wav_dir, bg.filename.values[0])  # 背景音文件路径
            # 混合后的文件名
            mixed_name = items['slice_file_name'][0:-4] + '_' + bg.scene_label.values[0] + '_' + bg.identifier.values[0] + '.wav'
            mixed_wav = os.path.join(fg_wav_dir, 'fold' + str(items['fold']), mixed_name)  # 混合后的保存路径，应该与前景音的fold保持一致
            # 混合后音频的输出路径
            audio_mix.mix_main(clean_file=fg_wav, noise_file=bg_wav, output_mixed_file=mixed_wav, snr=-5)
            # print("mixing ", index)
            mix_info = {
                "slice_file_name": [mixed_name],
                "fold": [items['fold']],
                "classID": [items["classID"]],
                "class": [items["class"]]
            }
            ub_leaking_nl = pd.concat([ub_leaking_nl, pd.DataFrame(mix_info)], ignore_index=True)
        print("mixed with ", scenes_items)

    print("car_horn accomplished")


def siren_aug(csv_dir):
    """
    用于对siren类的数据增强，这268个数据需要分别与DCASE四个scene的数据进行mix操作
    :param csv_dir: 获取ub_leaking_nl.csv路径: "./data/metadata/ub_leaking_nl.csv"
    :return:
    """
    global ub_leaking_nl
    siren_sample = make_dataframe.siren_sample_df(csv_dir)

    # print("car_horn_sample: ", car_horn_sample)
    for scenes_items in scenes:
        # items: park, public_square, street_pedestrian, street_traffic
        bg = DCASE2020_4scene.loc[DCASE2020_4scene["scene_label"] == scenes_items]  # 挑选了一个类别的背景音
        for index, items in siren_sample.iterrows():
            # index 范围：[0, 268]
            # items 内容：‘slice_file_name’, 'fold', 'classID', 'class'
            fg_wav = os.path.join(fg_wav_dir, 'fold' + str(items['fold']), items['slice_file_name'])  # 前景音文件路径
            bg = bg.sample(n=1)  # 注意bg是一个dataframe
            bg_wav = os.path.join(bg_wav_dir, bg.filename.values[0])  # 背景音文件路径
            # 混合后的文件名
            mixed_name = items['slice_file_name'][0:-4] + '_' + bg.scene_label.values[0] + '_' + bg.identifier.values[
                0] + '.wav'
            mixed_wav = os.path.join(fg_wav_dir, 'fold' + str(items['fold']), mixed_name)  # 混合后的保存路径，应该与前景音的fold保持一致
            # 混合后音频的输出路径
            audio_mix.mix_main(clean_file=fg_wav, noise_file=bg_wav, output_mixed_file=mixed_wav, snr=-5)
            # print("mixing ", index)
            mix_info = {
                "slice_file_name": [mixed_name],
                "fold": [items['fold']],
                "classID": [items["classID"]],
                "class": [items["class"]]
            }
            ub_leaking_nl = pd.concat([ub_leaking_nl, pd.DataFrame(mix_info)], ignore_index=True)
        print("mixed with ", scenes_items)

    print("siren_sample accomplished")


def other_class_aug(csv_dir):
    """
    用于对其余7个类的数据增强，这7个类，每个类需要抽取250个样例，与四个场景数据做mix操作
    :param csv_dir:获取ub_leaking_nl.csv路径: "./data/metadata/ub_leaking_nl.csv"
    :return:
    """
    global ub_leaking_nl
    other_class = make_dataframe.other_class_df(csv_dir)
    for classes in other_class['class'].unique():
        # classes内容：street_music, engine_idling, jackhammer, no_leaking, dog_bark, drilling, leaking
        some_class_sample = other_class.loc[other_class['class'] == classes].sample(n=250, replace=False)    # 挑选了一个类别,并采样
        for scenes_items in scenes:
            # items: park, public_square, street_pedestrian, street_traffic
            bg = DCASE2020_4scene.loc[DCASE2020_4scene["scene_label"] == scenes_items]  # 挑选了一个类别的背景音
            for index, items in some_class_sample.iterrows():
                # index 范围：[0, 250]
                # items 内容：‘slice_file_name’, 'fold', 'classID', 'class'
                fg_wav = os.path.join(fg_wav_dir, 'fold' + str(items['fold']), items['slice_file_name'])  # 前景音文件路径
                bg = bg.sample(n=1)  # 注意bg是一个dataframe
                bg_wav = os.path.join(bg_wav_dir, bg.filename.values[0])  # 背景音文件路径
                # 混合后的文件名
                mixed_name = items['slice_file_name'][0:-4] + '_' + bg.scene_label.values[0] + '_' + \
                             bg.identifier.values[0] + '.wav'
                mixed_wav = os.path.join(fg_wav_dir, 'fold' + str(items['fold']),
                                         mixed_name)  # 混合后的保存路径，应该与前景音的fold保持一致
                # 混合后音频的输出路径
                audio_mix.mix_main(clean_file=fg_wav, noise_file=bg_wav, output_mixed_file=mixed_wav, snr=-5)
                # print("mixing ", index)
                mix_info = {
                    "slice_file_name": [mixed_name],
                    "fold": [items['fold']],
                    "classID": [items["classID"]],
                    "class": [items["class"]]
                }
                ub_leaking_nl = pd.concat([ub_leaking_nl, pd.DataFrame(mix_info)], ignore_index=True)
            print("mixed with ", scenes_items)
        print("class {} accomplished".format(classes))
    print("data augmentation accomplished")


def data_augmentation(csv_dir):
    """
    9个类别的数据增强
    :param csv_dir: 获取ub_leaking_nl.csv路径: "./data/metadata/ub_leaking_nl.csv"
    :return:
    """
    car_horn_aug(csv_dir)
    siren_aug(csv_dir)
    other_class_aug(csv_dir)


if __name__ == "__main__":
    # data_augmentation("./data/metadata/ub_leaking_nl.csv")
    # ub_leaking_nl.to_csv("./data/metadata/ub_leaking_nl_aug.csv", index=False)
    # print(ub_leaking_nl.shape)
    # 现发现siren与leaking的classid重合，street_music与no_leaking的classid重合
    # 将siren的classID改为2，street_music的classID改为6
    data_aug = pd.read_csv("./data/metadata/ub_leaking_nl_aug.csv")
    # print(type(data_aug.loc[data_aug['class'] == 'car_horn'].classID.values[0]))
    # data_aug.loc[data_aug['class'] == 'siren', ['classID']] = 2
    # data_aug.loc[data_aug['class'] == 'street_music', ['classID']] = 6
    # data_aug.to_csv("./data/metadata/ub_leaking_nl_aug.csv", index=False)
    print("class siren: ", data_aug.loc[data_aug['classID'] == 9])

