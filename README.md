# 瓶子分割与分类

输入一张传送带大图，先用统一的单类 `bottle` 模型分割，按实例掩膜定位瓶子，再用可替换的分类模型判断类别。输出保留原图路径、瓶子编号、检测框、原尺寸掩膜、实际送入分类器的 PNG 和两项模型分数。业务判定由人工完成。

## 查看本轮结果

在已交付的工作树中打开 `artifacts/review/index.html`。点击大图可放大叠图，点击文件名可打开原图，各实例旁的链接对应其掩膜和分类输入。逐瓶结果在 `artifacts/review/instances.csv`，原图索引、模型 SHA-256、类别顺序及运行参数在 `artifacts/review/report.json`。图片和模型保存在本地。

本轮使用现有的 `hei_huizong_250428.onnx` 和分类训练目录的 `weights/best.pt`。439 张大图共得到 3,148 个实例，所有实例均取得分类结果。覆盖 23 个类别的 69 张真实分类小图上，原权重与导出 ONNX 的 Top-1 全部一致，类别概率最大绝对差为 0.00000233。使用同一旧分类器对照 C++ 时，11 张大图的 87 份掩膜、分类小图均逐像素相同。数据清洗和两类模型的训练入口均已准备；本次使用已有权重进行测试。

[C++ 评审、源码位置与双端对照](bottle_pipeline/CPP_REVIEW.md)

## 运行环境

以下是 PowerShell 命令，工作目录为仓库根目录。本机可直接沿用原 Python 3.10 环境。为了复现 C++ 的旋转矩形取整，本轮在工作树内独立使用 OpenCV 4.10。新克隆的仓库可在自己的虚拟环境运行 `python -m pip install -r requirements.txt`，并把下方的 `$py` 换为该环境的 Python 路径。

```powershell
Set-Location D:\Project\yolo26\worktree
$py = 'D:\Project\yolo26\venv\Scripts\python.exe'
$env:PYTHONPATH = (Resolve-Path .\artifacts\runtime).Path
```

## 准备训练数据

```powershell
& $py -m bottle_pipeline.prepare --seg D:\Data\seg_data\hei_huizong_0812 --cls D:\Data\clsData\huizon-xlj-0604 --output artifacts/datasets-v2
```

本机首轮交付的最终数据在 `artifacts/datasets`。再次处理时使用新的输出目录。默认种子是 42，按来源分组后分配约 80% / 10% / 10% 给训练、验证和测试。

- **分割。** 按文件主名配对 `JPEGImages` 和 `Annotations/*.json`，核对 LabelMe 的图名、尺寸、实例类别及多边形。把像素坐标转为 YOLO 的归一化多边形，每个瓶子一行 `0 x1 y1 x2 y2 ...`；明确标为无瓶子的图对应空标签。对最多越界半像素的点投影到图像边界，修正记录在清单中。无法确认配对或包含无效实例的整张图进入排除清单，以免把未标出的瓶子当作背景。
- **分类。** 读取分类目录下每个类名文件夹，检查图片能否完整解码；按解码像素去除同类重复，跨类同图的所有冲突副本都留给人工核查。能从文件名识别来源大图的样本按同一时间戳跨类别分组；其余只进入训练集。
- **追溯。** 两个任务各有 `manifest.csv`，包含来源、划分、目标文件、修正或排除原因；汇总位于 `summary.json`。训练使用的图像均为独立副本。分割目录提供 `data.yaml`、`images/{train,val,test}` 和对应的 `labels`；分类目录为 `{train,val,test}/类别名`。

| 本轮清洗      | 原始图片 |   训练 |  验证 |  测试 | 保留合计 |
| ------------- | -------: | -----: | ----: | ----: | -------: |
| 单类分割大图  |    8,530 |  6,241 |   746 |   761 |    7,748 |
| 23 类分类小图 |   46,939 | 29,676 | 3,493 | 3,808 |   36,977 |

分割保留 44,817 个瓶子标注，有 129 张保留图进行了亚像素边界修正。排除 779 张像素重复大图和 3 张图名错配标注。分类排除 9,368 张同类重复图和 594 张跨类标签冲突图，冲突涉及 180 组相同像素。另有 1,230 张保留小图无法从文件名追溯来源，均放在训练集，可识别的来源帧均只归入一个集合。上述“测试”列是新划分的训练数据留出集；人工复核用的 439 张大图来自单独的 `seg_testdata` 目录。

## 后续训练和导出

以下两条训练命令供后续使用，默认读取 `artifacts/datasets`。分割默认分辨率为 640，分类为 224。可用 `--datasets`、`--imgsz`、`--epochs`、`--batch`、`--device` 调整。新训练的权重进入 `artifacts/runs/<任务>/`。

```powershell
& $py -m bottle_pipeline.train segment --model D:\Project\yolo26\modelZoo\yolo26n-seg.pt
& $py -m bottle_pipeline.train classify --model D:\Project\yolo26\modelZoo\yolo26n-cls.pt
```

从已有权重导出 ONNX，可独立执行。导出时使用检查点的训练尺寸，启用动态批次，在临时目录中导出并完成一次 ONNX Runtime 推理后再写入目标文件。分割导出采用外部 NMS 的单类原始检测头和掩膜原型。

```powershell
& $py -m bottle_pipeline.export --weights D:\Project\yolo26\runs\classify\huizon-xlj-0604\weights\best.pt --output artifacts/models/classifier.onnx
# 有新的单类 bottle 分割权重后
& $py -m bottle_pipeline.export --weights artifacts/runs/segment/bottles/weights/best.pt --output artifacts/models/bottle.onnx
```

## 大图测试与更换分类模型

```powershell
& $py -m bottle_pipeline.predict --source D:\Data\seg_testdata --segmenter D:\Project\yolo26\modelZoo\hei_huizong_250428.onnx --classifier artifacts/models/classifier.onnx
```

`--source` 接受单张图或目录；对目录递归查找图片。默认在 `artifacts/predictions` 下创建新的运行目录，可用 `--output` 指定新目录。默认分割置信度 0.25、NMS IoU 0.45、掩膜阈值 0.5，可分别通过 `--confidence`、`--iou`、`--mask-threshold` 修改。

替换 `--classifier` 即可更换类别集合。读取 YOLO ONNX 内嵌的 `names` 类别顺序和输入尺寸，按模型生成相同尺寸的黑底分类画布。兼容固定或动态批次的 NCHW RGB 三通道分类器，输出须为单标签分类概率矩阵 `[batch, 类别数]`。若模型缺少类别元数据，提供 `--labels labels.txt`，文件使用 UTF-8，每行一个类别，按输出索引从 0 起排列；动态尺寸缺少元数据时使用 `--cls-size 高 宽`。若外部类别表与模型内嵌的类别顺序冲突，程序会报告错误。

分割统一使用单类 `bottle`。按 C++ 的实际行为保留瓶子方向和最小旋转矩形内部像素，分类用的小图可能包含矩形内的背景。置信度是模型输出，人工检查时可结合原图判断漏检、重叠或分类错误。
