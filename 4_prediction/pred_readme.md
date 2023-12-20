## Argoverse数据整理
1. `parquet`(agent)信息
    
2. 读取`json`(地图元素)
    `drivable_areas`,`lane_segments`,`pedestrian_crossings`
    - `polygon`:针对lane_segment,polygon在centerline的起点;如果是人行横道,polygon有两个,分别是edge0和edge1的起始和结束的中点
    > `position`<br> `orientation`<br> `height`<br> `type`:vehicle,bus,bike,pedestrian<br> `is_intersection`<br>
   `polygon_to_polygon_edge_index`:记录lane_segment之间的前驱后继关系.和左右链接关系<br>
    `point_to_polygon_edge_index`:记录`point`与`polygon`之间的关系
    - `point`
   > `position`: 包括左边界,右边界和中心线<br> `orientation`<br> `magnitude`:[矩阵Frobenius范数](https://yiyan.baidu.com/share/DMo3p1QWpN)<br> `height`<br> `type`:道路线类型<br> `side`:左右以及中心线
3. 整理并保存为pkl文件


HeteroData:异构图

- 节点: [`agent`,`map_polygon`,`map_point`]
- 边: [`map_point`->`map_polygon`, `map_polygon`->`map_polygon`]
- stores:列表
    - 0 :存储4个场景id和城市(`batch_size`)
    - 1 : 存储`agent`节点信息
      > `num_nodes`: agent总数<br> `av_index`:av对应的索引<br> `valid_mask`<br>
      `predict_mask`<br> `id`<br> `type`<br> `category` <br> `position` <br> `heading`<br> `velocity`<br>
      `batch` <br> `ptr`<br>
    - 2: 存储`map_polygon`节点信息
      > `num_nodes`:节点个数<br> `position`:polygon位置<br> `orientation`<br>
      `height`<br> `type`<br> `is_intersection`<br> `batch` `ptr`
    - 4 ~ 5: 存储两个边信息

图网络的attention机制
Fourier变换

## QCNet:
### 1. Encoder
#### 1.1 MapEncoder
- FourierEmbedding 
#### 1.2 AgentEncoder

### 2. Decoder