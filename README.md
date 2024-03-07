**状态:** 存档(代码按原样提供,不再更新)

# gpt-2

这是论文["语言模型是无监督的多任务学习者"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)中的代码和模型。

你可以在我们的[原始博客文章](https://openai.com/research/better-language-models/)、[6个月后的跟进文章](https://openai.com/blog/gpt-2-6-month-follow-up/)和[最终文章](https://www.openai.com/blog/gpt-2-1-5b-release/)中阅读关于GPT-2及其分阶段发布的内容。

我们还[发布了一个数据集](https://github.com/openai/gpt-2-output-dataset)供研究人员研究它们的行为。

<sup>*</sup> *注意,我们之前的参数计数是错误的(在我们之前的博客文章和论文中)。因此你可能看到小型模型被称为117M,中型模型被称为345M。*

## 使用方法

该存储库旨在为研究人员和工程师提供一个起点,以试验GPT-2。

有关基本信息,请参阅我们的[模型卡片](./model_card.md)。

### 一些警告

- GPT-2模型的稳健性和最坏情况行为尚不清楚。与任何机器学习模型一样,如果在没有微调或在可靠性很重要的安全关键应用中使用时,请仔细评估GPT-2是否适合您的用例。
- 我们训练GPT-2模型的数据集包含许多带有[偏差](https://twitter.com/TomerUllman/status/1101485289720242177)和事实错误的文本,因此GPT-2模型也可能存在偏差和不准确。
- 为避免样本被误认为人工书写,我们建议在广泛传播之前明确标记样本为合成的。我们的模型经常在细微的方面缺乏连贯性或准确性,需要人类仔细阅读才能注意到。

### 与我们合作

如果您正在从事有趣的研究或正在开发GPT-2的应用,请[告诉我们](mailto:languagequestions@openai.com)!我们特别有兴趣听取并可能与那些研究以下领域的人合作:
- 潜在的恶意用例及其防御措施(例如检测合成文本的可能性)
- 模型中存在的有问题内容(例如偏差)的程度以及有效的缓解措施

## 开发

请参阅[DEVELOPERS.md](./DEVELOPERS.md)

## 贡献者

请参阅[CONTRIBUTORS.md](./CONTRIBUTORS.md)

## 引用

请使用以下bibtex条目:
```
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}
```

## 未来工作

我们可能会发布用于在各种基准测试上评估模型的代码。

我们仍在考虑发布更大的模型。

## 许可证

[修改后的MIT](./LICENSE)