# 请求调度预实验（未写入论文）

本目录保存 DC-EDRN seed42 冻结权重的请求队列预实验。实验没有训练、阈值重选、权重修改或 GPU 测量；它只用于判断 ISPA Track 2 的调度与资源管理方向是否值得继续补测。

## 已运行内容

- 设备：CPU，4 个 PyTorch 线程；当前机器的 Conda PyTorch 没有 CUDA。
- 模型：`DC_EDRN:42`，即已有 `Paper_NoExplicitDuration_seed42.pth`。
- 服务测量：batch 1/4/8/16/32，每档 8 次，预热 2 次。
- 请求轨迹：96 个请求，周期到达率 5、10、20、40 texts/s。
- 策略：逐请求；batch 8、最长等待 20 ms；batch 32、最长等待 5 ms。

## 初步结果

在 40 texts/s 的虚拟到达率下：

- 逐请求：P95 约 14.05 s，吞吐约 5.60 texts/s；
- batch 8 / 20 ms：P95 约 2.33 s，吞吐约 20.63 texts/s；
- batch 32 / 5 ms：P95 约 0.74 s，吞吐约 33.14 texts/s。

在 20 texts/s 下，逐请求 P95 约 11.79 s，batch 8 / 20 ms 约 0.74 s，batch 32 / 5 ms 约 0.75 s。

这些数字来自测得的 CPU batch 服务时间和确定性虚拟队列，不是多进程、多 GPU、集群或 Triton 实际服务的测量。它们说明动态批处理值得进一步做正式实验，但不能直接加入论文作为分布式性能结论。

原始完整记录：`queue_scheduling_pilot_full.json`。短测记录：`queue_scheduling_pilot.json`。运行脚本：`..\\tools\\queue_scheduling_pilot.py`。
