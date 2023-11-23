# exploring_data_driven
## 1. basic
### 1.1 [pytorch](https://yiyan.baidu.com/share/REMpDKUfCt)与线性回归问题
- pytorch特点
> [动态计算图](https://zhuanlan.zhihu.com/p/598760275)<br>
> [张量计算](https://blog.csdn.net/qq_42681787/article/details/129323096) <br>
>高效的gpu加速 <br>
> [自动求导](https://blog.csdn.net/Xixo0628/article/details/112669929?spm=1001.2014.3001.5502) <br> 广泛的预训练模型库 <br> 强大的线性代数支持<br> 广泛的社区支持
- Tensor创建与操作
- 线性回归
### 1.2 workflow
#### 1.2.1 数据
- 数据集

|                             数据集                              |                                        文件                                        |                                        benchmark                                         |
| :-------------------------------------------------------------: | :--------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: |
|           [MNIST](http://yann.lecun.com/exdb/mnist/)            |                                                                                    |                                                                                          |
| [FashinMNIST](https://github.com/zalandoresearch/fashion-mnist) | [benchmark](https://paperswithcode.com/sota/image-classification-on-fashion-mnist) |                                                                                          |
|     [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)      |                                                                                    |      [benchmark](https://paperswithcode.com/sota/image-classification-on-cifar-10)       |
|          [COCO](https://cocodataset.org/#format-data)           |                  [file](/home/renjie/renjie_ws/dataset/COCO2017)                   |          [benchmark](https://paperswithcode.com/sota/object-detection-on-coco)           |
|        [kitti](https://paperswithcode.com/dataset/kitti)        |                                                                                    |                                                                                          |
|              [nuscenes](https://www.nuscenes.org/)              |           [file](/home/renjie/renjie_ws/dataset/nuscenes/data/v1.0-mini)           | [benchmark](https://paperswithcode.com/sota/3d-object-detection-on-nuscenes-camera-only) |

- [Dataset & Dataloader](https://yiyan.baidu.com/share/CBuUUXkDTU)
> 从库(`torchvision`,`torchaudio`,`torchtext`)中加载数据集<br>
> [自定义数据集](https://blog.csdn.net/qianbin3200896/article/details/119832583?spm=1001.2014.3001.5501#t9)<br>

- [transform](https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html)
> 数据清洗、标准化、归一化、填充缺失值、转换数据类型等<br>
> [图像常见transform](https://pytorch.org/vision/stable/transforms.html)

#### 1.2.2 模型
- [线性(linear)](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)

- [激活函数(activation)](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
> [为什么要激活函数]( https://yiyan.baidu.com/share/ZCqwR1fAGj )<br>
> [常见激活函数](https://zhuanlan.zhihu.com/p/352668984)<br>
> [什么是梯度消失与爆炸](https://zhuanlan.zhihu.com/p/483651927)<br>
> [什么是ICS](https://yiyan.baidu.com/share/GHpggdsBEy )

|  激活函数   |   输出值范围   |     导数值范围      |                 优点                  |               缺点                |
| :---------: | :------------: | :-----------------: | :-----------------------------------: | :-------------------------------: |
|  `sigmoid`  |     (0,1)      |      (0, 0.25)      |               符合概率                | 梯度消失<br> 非0均值,破坏数据分布 |
|   `tanh`    |     (-1,1)     |       (0, 1)        |                 0均值                 |            易梯度消失             |
|   `ReLU`    |    (0,inf)     |  0 x<=0;<br> 1 x>0  |            计算速度块<br>             |    0处不可微 ;<br>神经元"死亡"    |
| `LeakyReLU` | (-eps*inf,inf) | eps x<=0;<br> 1 x>0 | 避免梯度方向锯齿问题,参数更新更加平滑 |                                   |
|   `PReLU`   |  (-P*inf,inf)  |  P x<=0;<br> 1 x>0  |                                       |                                   |
|   `RReLU`   |                |                     |                                       |                                   |


- [归一化(Normalization)](https://pytorch.org/docs/stable/nn.html#normalization-layers)
> [常见归一化层](https://blog.csdn.net/weixin_43570470/article/details/124043037)<br>
> [归一化通俗理解]( https://yiyan.baidu.com/share/IybijX30zX) <br>
> **batch_norm**  **layer_norm** instance_norm group_norm
- [池化层(pooling)](https://pytorch.org/docs/stable/nn.html#pooling-layers)
> [常见池化层图示](https://zhuanlan.zhihu.com/p/77040467)<br>
> [池化层通俗理解]( https://yiyan.baidu.com/share/9GywU6CZXG)

- [dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout)
> [dropout图示](https://zhuanlan.zhihu.com/p/390990848?utm_id=0)<br>
> [将dropout理解为集成学习]( https://yiyan.baidu.com/share/63qP4PxYQ0 )

- 卷积(convolution)
> [普通卷积图示](https://blog.csdn.net/m0_47005029/article/details/129270974)<br>
> [各式卷积](https://blog.51cto.com/u_14439393/5945930) [各式卷积](https://blog.csdn.net/m0_62919535/article/details/131317667?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-131317667-blog-129270974.235%5Ev38%5Epc_relevant_anti_t3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-131317667-blog-129270974.235%5Ev38%5Epc_relevant_anti_t3&utm_relevant_index=6)<br>
> 怎么实现高效卷积计算

- attention
- transformer

        attention
        deformable attention


#### 1.2.3 损失函数

#### 1.2.4 优化器

#### 1.2.4 训练迭代,加载,保存和可视化

### 1.3 [quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

## 2. 分类任务
## 3. 目标检测、语义分割和实例分割
## 4. Lidar感知与预测模型
## 5. BEV
## 6. PNC和E-E探索