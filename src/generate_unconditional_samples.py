#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def sample_model(
    model_name='124M',
    seed=None,
    nsamples=0,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
):
    """
    运行 sample_model 函数
    :model_name='124M' : 字符串，指定要使用的模型
    :seed=None : 随机数生成器的整数种子，固定种子以重现结果
    :nsamples=0 : 要返回的样本数量，如果为0，则持续生成样本
    :batch_size=1 : 批处理大小（仅影响速度/内存）
    :length=None : 生成文本中的标记数，如果为None（默认），则由模型超参数确定
    :temperature=1 : 控制玻尔兹曼分布中的随机性的浮点值。较低的温度导致更少的随机完成。当温度接近零时，模型将变得确定性和重复。较高的温度导致更多的随机完成。
    :top_k=0 : 控制多样性的整数值。1表示每个步骤（标记）仅考虑1个单词，导致确定性完成，而40表示每个步骤考虑40个单词。0（默认）是一个特殊设置，表示没有限制。40通常是一个不错的值。
    :models_dir : 包含模型子文件夹的父文件夹的路径（即包含 <model_name> 文件夹）
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx
    elif length > hparams.n_ctx:
        raise ValueError("无法生成比窗口大小 %s 更长的样本" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        np.random.seed(seed)
        tf.set_random_seed(seed)

        output = sample.sample_sequence(
            hparams=hparams, length=length,
            start_token=enc.encoder['<|endoftext|>'],
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )[:, 1:]

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        generated = 0
        while nsamples == 0 or generated < nsamples:
            out = sess.run(output)
            for i in range(batch_size):
                generated += batch_size
                text = enc.decode(out[i])
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)

if __name__ == '__main__':
    fire.Fire(sample_model)