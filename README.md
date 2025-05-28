我们在 MMpose 框架的基础上进行了函数扩展与修改，以实现并验证我们提出的方法。具体修改如下：# POPR

heatmap_head.py
修改了关键点检测头的结构，以支持自定义的预测机制和输出格式。我们在此基础上集成了提出的 POPR 正则策略。

heatmap_loss.py
实现了一个包含序保持正则项（POPR）的新损失函数，用于替代或补充标准的 MSE 损失函数，适用于热图回归任务。

msra_heatmap.py
修改了目标热图的生成逻辑。在原始高斯热图生成的基础上，扩展支持多变量热图分布，并计算热图的期望坐标以服务于 POPR 损失函数。

所有修改均与 MMpose 原有的训练和评估流程兼容。
请确保相应的配置文件（config）已更新以调用新实现的模块。
将上述 Python 文件分别放置在 MMpose 工程的相应目录中：
heatmap_head.py：放在 mmpose/models/heads/
heatmap_loss.py：放在 mmpose/models/losses/
msra_heatmap.py：放在mmpose/codecs/
