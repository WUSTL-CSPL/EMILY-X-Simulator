# EMILY 定位数据集

这个目录整理了当前生成数据集所需的代码和说明，可以独立作为 GitHub 仓库。
原来的工作目录、数据和脚本没有搬走。

## Greg 的代码怎么办？

`Incident_simulator/` 是 **Git submodule（子模块）**。我们的仓库记录 Greg
仓库的地址和固定的 commit，下载时 Git 再拉取他的代码。
不需要你有他的写权限，也不会向他的仓库提交任何修改。

当前固定在 `a6124824f653467173c845c4ff609d4d35d6902f`，对应已验证的 v3.2。
Greg 更新 main 后，这里不会自动换版本，避免模拟结果悄悄改变。

## 在这台机器上运行

进入这个新目录，然后：

```bash
python3 scripts/link_local_data.py --source ..
bash setup_simulator.sh
.venv/bin/python scripts/check_inputs.py --reference-v3
```

第一步复用原项目的 IQ 和地形，不复制大文件。其他机器需要自己准备这些数据。
完整生成命令见 [英文 README](README.md#generate-the-v3-dataset)。

数据流程是：

1. 在 20 × 20 km 区域里分格随机生成 4,000 个发射机位置。
2. 每个位置模拟 60 秒，两个固定接收器产生 EMILY incidents。
3. 按不同地理区域选出 1,000 个 train、200 个 validation、200 个 test。

每个样本把两个接收器的 observations 放在一起，目标标签是发射机坐标。
incident 数量不等于样本数量。现在只有一种 LTE 发射机，主要用于定位实验。

## 哪些内容会上传？

- 上传：脚本、文档、依赖版本、参考哈希，以及 Greg 仓库的子模块引用。
- 本地保留：IQ、SRTM 地形、生成的数据集、运行日志、虚拟环境。

`.gitignore` 已配置好。当前还没有创建 GitHub 远程仓库或 push。
在 GitHub 创建一个空仓库后，按 [发布步骤](docs/GITHUB.md#publish-this-folder) 操作。
其他人用 `git clone --recurse-submodules ...` 下载即可一并获得 Greg 的代码；
IQ 和地形仍需另外准备。
