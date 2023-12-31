# Introduction

## Speck

Speck™是全球首款“感算一体”动态视觉智能SoC。实现了基于异步逻辑范式的大规模脉冲卷积神经网络（sCNN）芯片架构。芯片最多可配置高达32万脉冲神经元并且芯片内部集成了动态视觉传感器（DVS）为其提供实时高效的动态视觉输入。Speck™创举性的集动态视觉传感和事件驱动运算于一体， 提供高度实时、集成和低功耗的动态视觉解决方案，在传统应用场景下，Speck™可在mW级的超低功耗、ms级的实时响应下，完成智能场景分析。可以广泛应用于移动设备、物联网（IoT）、智能家电、智能安防、智慧养殖等领域。

## Speck devkit

Speck devkit 是SynSense时识科技最新款类脑视觉开发套件，全新升级的Speck™系列SoC集成 DVS动态视觉传感器/事件相机，大幅度提升了感光性能并降低了噪音，同时，SoC片内集成32万脉冲神经元类脑智能处理器，将动态实时图像处理能力的功耗保持在亚毫瓦量级。利用SynSense时识科技开源库Sinabs和工具链Samna，可轻松使用片上DVS传感器并搭建高达9层的脉冲卷积神经网络。

Speck™开发套件为SynSense时识科技目前在售的高性能类脑智能视觉开发平台，支持实时监测、手势识别、行为检测、光流、跟随等多场景复杂视觉类应用模型开发。



# Main features

- 1个内置的DVS层
- 9个卷积层（卷积层含Pooling）
- 1个读出层
- SPI 主/从接口，用来配置
- Pooling: {1:1, 1:2, 1:4}
- 多扇出操作，最大为2

## DVS Layer

- 128x128 像素阵列
- 噪声过滤
- DVS事件点记性可调
- ROI
- 水平/竖直镜像
- 90 degree 翻转步长
- 动态范围：不小于80dB

## Conv Layer

- 9 layers
- 权重精度：8
- 神经元状态精度：16
- 最大卷积核尺寸：16x16
- 卷积/池化步长：{1, 2, 4, 8}
- 卷积/池化补0: [0:7]
- Pooling: {1:1, 1:2, 1:4}
- 多扇出操作，最大为2
- 卷积输出可降采样
- 卷积层通信拥塞处理（这个很重要）
- 第0层和第1层卷积层支持并行数据处理

## 读出层

- 最大16分类输出（注意这里是15个有效+1个无效）
- 滑动均值步长{1, 16, 32}
- 4种读出{无效类，阈值类，极值类，特定类}
- 这里可以通过4位可编码输出与1位中断，直接输出极值或阈值分类

## 内部慢速时钟

内部使用慢速时钟来支持许多基于时间周期的功能。

- 泄漏时钟，CNN 层包括泄漏操作，可以根据时钟设置对所有配置的神经元状态进行操作。
- DVS预处理滤波器：DVS滤波器使用慢时钟作为时间参考来更新其内部状态
- 读出层：读出层使用慢时钟周期作为移动平均步长来计算移动平均线。

# Configration

| Pin No. | Pin Name    | Usage                         |
| :-----: | ----------- | ----------------------------- |
|    1    | SPI_M_CSN   | 输入，连接Flash               |
|    2    | CLK_SLOW    | 输入，用作READOUT参考时间窗   |
|    3    | MODE_SEL    | 配置输入                      |
|    4    | INTERRUPT_O | 输出，芯片有效输出标志位      |
|    5    | RSV1        | 预留                          |
|    6    | RSV2        | 预留                          |
|    7    | GND         | GND                           |
|    8    | VDD_IO      | 电源输入，芯片IO电源供应      |
|    9    | VDD_RAM     | 电源输入，芯片内ram电源供应   |
|   10    | VDD_LOGIC   | 电源输入，芯片内logic电源供应 |
|   11    | VDD_D_PIXEL | 电源输入，DVS阵列数字电源供应 |
|   12    | GND         | GND                           |
|   13    | GND         | GND                           |
|   14    | VDD_A_PIXEL | 电源输入，DVS阵列模拟电源供应 |
|   15    | GND         | GND                           |
|   16    | READOUT1    | READOUT1                      |
|   17    | READOUT3    | READOUT3                      |
|   18    | READOUT4    | READOUT4                      |
|   19    | READOUT2    | READOUT2                      |
|   20    | CLK_I       | 输入，芯片SPI等接口时钟源     |
|   21    | RB_I        | 输入，系统复位，低有效        |
|   22    | SPI_M_MOSI  | 输入，SPI- master，连接Flash  |
|   23    | SPI_M_MISO  | 输入，SPI- master，连接Flash  |
|   24    | SPI_M_CLK   | 输入，SPI- master，连接Flash  |
|  25-28  | GND         | 接地                          |

## Memory capacity

Speck由9层可配置的conv layer组成，但是每层的容量有限，所以配置之前需要计算好内存容量。如果内存过大，配置过程中会在仿真这一步报错，即“找不到可用的配置顺序”。

| Core | Kernel memory | Neuron memory |
| :--: | :-----------: | :-----------: |
|  0   |     16 Ki     |     64 Ki     |
|  1   |     16 Ki     |     64 Ki     |
|  2   |     16 Ki     |     64 Ki     |
|  3   |     32 Ki     |     32 Ki     |
|  4   |     32 Ki     |     32 Ki     |
|  5   |     64 Ki     |     16 Ki     |
|  6   |     64 Ki     |     16 Ki     |
|  7   |     16 Ki     |     16 Ki     |
|  8   |     16 Ki     |     16 Ki     |

详细计算方式可以参考官方文档给出的内存计算方式，实际发现有点误差，但是八九不离十。



























