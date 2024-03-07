#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def interact_model(
    model_name='124M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
):
    """
    交互式地运行模型
    :model_name=124M : String, 选择要使用的模型
    :seed=None : 随机数生成器的整数种子，固定种子以重现结果
    :nsamples=1 : 要返回的样本总数
    :batch_size=1 : 批次数（仅影响速度/内存）。必须能整除 nsamples。
    :length=None : 生成文本中的令牌数，如果为 None（默认），则由模型超参数确定
    :temperature=1 : 控制玻尔兹曼分布中的随机性的浮点值。较低的温度导致更少的随机完成。当温度接近零时，模型将变得确定性和重复。较高的温度导致更多的随机完成。
    :top_k=0 : 控制多样性的整数值。1 表示每步（令牌）只考虑 1 个词，导致确定性完成，而 40 表示每步考虑 40 个词。0（默认）是一个特殊设置，表示没有限制。40 通常是一个不错的值。
    :models_dir : 包含模型子文件夹的父文件夹的路径（即包含 <model_name> 文件夹）
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("无法生成比窗口大小 %s 更长的样本" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        while True:
            raw_text = input("模型提示 >>> ")
            while not raw_text:
                print('提示不能为空！')
                raw_text = input("模型提示 >>> ")
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " 样本 " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)

if __name__ == '__main__':
    fire.Fire(interact_model)
