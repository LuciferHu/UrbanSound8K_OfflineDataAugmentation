# -*- coding:utf-8 -*-
import pandas as pd
"""
本文件用于准备好所需要到的dataframe。需要读取的文件有：
* 源音ub_leaking_nl.csv，路径：/home/richardhu/Documents/PycharmProjects/ub_leaking_with_augmentation/data/metadata
* 背景音meta.csv，路径：/home/richardhu/Downloads/DCASE2020/train
"""


def ub_leaking_nl_df(data_dir):
    return pd.read_csv(data_dir)


def DCASE2020_4scene_df(data_dir):
    DCASE2020 = pd.read_csv(data_dir, sep='\t')
    DCASE2020_4scene = DCASE2020.loc[(DCASE2020["scene_label"] == "park") |
                                     (DCASE2020["scene_label"] == "public_square") |
                                     (DCASE2020["scene_label"] == "street_pedestrian") |
                                     (DCASE2020["scene_label"] == "street_traffic")]
    return DCASE2020_4scene


def car_horn_sample_df(data_dir):
    """
    car_horn类别需要抽取393个文件
    :param data_dir: 路径
    :return: 
    """
    ub_leaking_nl = ub_leaking_nl_df(data_dir)
    car_horn = ub_leaking_nl.loc[ub_leaking_nl["class"] == 'car_horn']
    car_horn_sample = car_horn.sample(n=393, replace=False)
    return car_horn_sample


def siren_sample_df(data_dir):
    """
    siren类需要抽取268个文件
    :param data_dir: 路径
    :return:
    """
    ub_leaking_nl = ub_leaking_nl_df(data_dir)
    siren = ub_leaking_nl.loc[ub_leaking_nl["class"] == 'siren']
    siren_sample = siren.sample(n=268, replace=False)
    return siren_sample


def other_class_df(data_dir):
    """
    其余拥有1000个示例的类别
    :param data_dir:
    :return:
    """
    ub_leaking_nl = ub_leaking_nl_df(data_dir)
    other_class = ub_leaking_nl.loc[(ub_leaking_nl['class'] != "car_horn") &
                                    (ub_leaking_nl['class'] != "siren")]
    return other_class


if __name__ == "__main__":
    print(car_horn_sample_df("./data/metadata/ub_leaking_nl.csv"))