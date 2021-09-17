# 成为 PyMilvus 的贡献者吧！

PyMilvus 是 Milvus 的 Python SDK，属于 Milvus 社区的开源项目之一，是 Milvus 向外辐射出来最受欢迎的项目。PyMilvus 从诞生之初就一直伴随 Milvus 的成长而成长，目前在 Github 上有 300 多颗星，贡献者有40 人 （9.10.2021）。

很多对 Milvus 感兴趣的人就是通过使用 PyMilvus 来第一次接触 Milvus。

PyMilvus 拥有一个长期维护的 1.x 版本，兼容 Milvus 1.x。目前活跃开发的主分支是 PyMilvus 2.x 版本，兼容的是 Milvus 2.x。

Milvus 社区的所有项目都非常欢迎大家参与贡献、共建社区，而 PyMilvus 不同于 Milvus 社区其他项目的地方有：

- 纯 Python；
- 支撑着 Milvus 的 E2E 测试和 Benchmark 以及 Milvus Bootcamp 项目；
- 作为 Python 库，使用场景更多；

真诚的欢迎大家来参与 PyMilvus 项目，共建一个协作、开源、开放的社区。如果你已经非常熟悉 PyMilvus 的代码和使用方式，非常欢迎大家来回馈社区，帮助更多的社区新人，将开源、协作、开放的精神传递下去。

## 贡献从这里开始

PyMilvus 的 Github issue 列表中，打上了 [good-first-issue](https://github.com/milvus-io/pymilvus/labels/good%20first%20issue) 和 [help-wanted](https://github.com/milvus-io/pymilvus/labels/help%20wanted) 标签的都是入门级别的 issue。如果你还在熟悉项目，这些 issue 是极好的出发点。

如果你想挑战一下，不妨看看拥有 [Hacktoberfest](https://github.com/milvus-io/pymilvus/labels/Hacktoberfest) 标签的 issue。

## 我可以贡献什么？

如果你发现任何问题，你可以：

- 提 issue 指出问题是什么
- 在 issue 中给出最小复现方法 （可选）
- 在 issue 中给出解决方案 （可选）
- 提 PR 修复这个 issue （可选）

如果你对已经存在的问题感兴趣，你可以：

- 在标有 `question` 标签的 issue 内回答问题帮助他人
- 在标有 `bug`、`improvement` 和 `enhancement` 标签的 issue 内帮助他人：
  - 提问、复现、给出解决方案
  - 提 PR 修复这个 issue

如果你想让 PyMilvus 拥有新功能，你可以：

- 提 issue 指出你想要的新功能以及原因
- 在 issue 中指出这个功能的实现方案以及测试大纲 （可选）
- 提 PR 实现这个新的功能 （可选）

如果你对已经存在的 PR 感兴趣，你可以：

- 参与代码 review，并给出合适意见
- 指导社区新人走完 PR 的流程

这里提到的问题、新功能、疑问不仅指 Python 代码，还包括各种文档（技术文档，API 参考手册，贡献文档等等）。

## PyMilvus 代码结构

`docs/`: 除了 API Reference 外的文档源码所在地，使用 sphinx 组织编译，大部分文档内容都在 `docs/source` 目录下的 rst 文件中。

`examples/`: 包含一些可以直接运行的 python 脚本，通过例子介绍 PyMilvus 每个接口的使用方法。

`pymilvus/`：PyMilvus 的源码目录。

`tests/`: 单元测试目录。

`CONTRIBUTING.md`: 本文档。

`LICENSE`: PyMilvus 项目遵循的开源协议。

`Makefile`: 方便 Github action 运行的脚本。

`OWNERS`: 这个文件指定了当前目录的 reviewers 和 approvers。他们的人选是根据贡献者的活跃度和对当前代码贡献量综合敲定的。活跃的 contributors 会被列为 reviewers，承担起审核代码的责任。当一个 reviewer 审核了一段时间代码且在社区中一直处于活跃状态时，reviewer 会被列为 approver，负责对 PR 除了代码外的内容审核。如果你提了一个 PR 而不知道应该给谁审核，可以从这里面指定的贡献者挑选 reviewer 和 approver 来审核 PR。

`README.md`: Readme 文档。

`requirements.txt`: 开发 PyMilvus 时依赖的第三方库。

`setup.py`: PyMilvus 的打包脚本

## 贡献前必知

### PyMilvus 的 Github 工作流

## 提 PR 前必知

### 如何写 commit message ？

### 签署 DCO

## 合并 PR

### reviewer 审阅 PR 的哪些地方

### approver 审阅 PR 的哪些地方

### 通过所有 Github Actions

## 恭喜你！你已经成为了 Milvus 社区的贡献者！

除了和代码、机器打交道，你还可以和 Milvus 社区中的人交流。社区中每天都有很多新面孔加入，当他们遇到的困难正好是你所了解的地方，请尽情的帮助这些人。回想你初次接触 Milvus 接受过的帮助，你也可以将这样的交流互助精神不断传递下去，我们一起共创一个协作、开源、开放、包容的社区。









