## FasterRCNN
### AnchorGenerator
1. 锚框生成器. 指定长度比和size
2. `sizes` 和 `aspect_ratios` 个数和 feature maps个数相同
3. `generate_anchors`

### RPNHead

### BoxCoder
1. decode : 从anchors和编码的相对box offset中获取 解码的boxes
获取 (x1, y1, x2, y2)

### Matcher
1. 根据`high_threshold (IOU 0.7)`和`low_threshold (IOU 0.3)`将anchors 和 targets分为3个level
-   matches >= high_threshold
-   BETWEEN_THRESHOLDS
-   BELOW_LOW_THRESHOLD
2. 根据 `match_qualityu_matrix`

### BalancedPositiveNegativeSampler 
1. batch_size_per_image:256  在rpn阶段采样256个样本 
2. positive_fraction:0.5     其中正样本占0.5
3. 根据 matched_indx 的值,确定正样本和负样本的index
4. 随机选择 num_pos 个正样本 和num_neg 个负样本
5. 

### RegionProposalNetwork (RPN) 区域建议网络
objectness [Tensor(2, 15, 25, 39)]
pred_bbox_deltas [Tensor(2, 60, 25, 39)]
anchors [Tensor(14625, 4), Tensor(14625, 4)]

num_anchors_per_level_shape_tensors [ Tensor(15, 25, 39) ]
num_anchors_per_level [ 14625 ]

`concat_box_prediction_layers`: 
objectness [Tensor(29250, 1)]
pred_bbox_deltas [Tensor(29250, 4)]

proposals  [Tensor(29250, 1 , 4)] 根据`pred_bbox_deltas`和`anchors`获取 解码每个anchor对应的box (x1, y1, x2, y2)
proposals  [Tensor(2, 14625 , 4)]  (imgs, anchors, 4)

`filter_proposals`: 根据 proposals objectness, image_shapes, num_anchors_per_level
1. 获取objectness前2000个的索引
     根据 objectness 排名,获取前2000个对应的索引
     根据 top_n_index 获取 对应的objectness, levels, proposals
     将 objectness 转化成概率
2. 根据image_size裁剪boxes
    `clip_boxes_to_image` 根据 图片大小裁剪box, 删除小的boxes
3. 删除概率值低于`score_thresh`(0.0)的box
4. 非极大值抑制NMS(`nms_tresh`(0.7)) 
5. 在nms过后,保留`post_nms_top_n`(2000)
6. 最后返回不同图像的boxes,scores [ Tensor(remain_box, 4),...]  [ Tensor(remain_box, 1),...]

boxes 最后保留的不同图像的boxes [Tensor(1539, 4), Tensor(1702, 4)]
scores 最后保留的不同图像的scores [Tensor(1539), Tensor(1702)]

`assign_targets_to_anchors`:
anchors_per_image 每个图片的anchors Tensor(14625, 4)
targets_per_image 每个图片的标签 {"boxes":, "labels": }
match_quality_matrix 每个anchors与target的iou  Tensor(2,14625)
matched_idxs 根据`Matcher`获得的匹配等级(-2:低  -1: BETWEEN_THRESHOLDS 其他对应的target 的id)
matched_gt_boxes_per_image 每个anchor匹配的boxes
labels_per_image 每个anchor对应是否有正标签
labels_per_image 设置 负样本(background)为0 , 丢弃的设置为-1

labels rpn样本标签 [ tensor(14625), tensor(14625)] 取值(-1:丢弃 0:负样本 1:正样本)
matched_gt_boxes  rpn回归的标签,每个anchor对应一个target box [tensor(14625, 4), tensor(14625, 4)]

regression_targets 根据 matched_gt_boxes 和anchor ,计算编码值


`compute_loss`
调用`BalancedPositiveNegativeSampler` 获取适量的正样本和负样本的index
针对 分类损失 : 考虑正负样本
针对 回归损失: 只有正样本

1. rpn head  输入backbone输出的feature,
    - 输出 [bs, num_anchors*4, feature_h, feature_w]的 box回归值(编码后的值)
    - 输出 [bs, num_anchors, feature_h, feature_w]的类别

2. anchor_generator 输入图像和features
    - 输出 anchor,每个feature对应的位置有 num_scale*num_aspect_ratio(5*3)个anchor,所以共有 num_scale*num_aspect_ratio*feature_h*feature_w

3. reshape objectness 和 pred_bbox_deltas, 使得 
   - 输出 objectness  (imgs*mul_feature_pos,1)
   - 输出 pred_bbox_deltas (imgs*mul_feature_pos,4)

4. 将 rpn网络输出的 pred_bbox_deltas box编码值按照anchor进行解码
    - 输出 proposals
5. 过滤 proposal 给下游roi 网络
    - 先获取概率前pre_nms_top_n(2000)个的索引
    - 根据图像信息,裁剪box,然后过滤small box
    - 删除low scoring 的box
    - 根据nms(0.7)进行过滤
    - 最后取概率前post_nms_top_n(2000)个的索引

6. 将 标签 指派给各anchor
    - 计算 标签box与各个anchor的交并比 IOU
    - 利用Matcher中指定的IOU阈值,得到每个anchor的匹配target, 有以下可能(-2:低  -1: BETWEEN_THRESHOLDS 其他对应的target 的id)
    - 整理获得在rpn各个anchor对应labels(discard,fg,bg) 和每个anchor应该对应目标框  matched_gt_boxes

7. 将各个anchor对应的matched_gt_boxes 编码..这里出现大量不必要计算.因为matched_gt_boxes很多不需要用到,只有和target box的 IOU大于0.7,下游才用到

8. 计算loss
    - 根据前景和背景进行采样,尽量使正负样本个数相当
    - 针对回归问题: 对正样本进行smooth_l1_loss计算
    - 针对分类问题: 对正负样本进行二分类交叉熵计算

### ROIHead

`select_training_samples`
1. 将gt bbox 加到 proposal
2. `assign_targets_to_proposals`  获得每个proposal所对应的分类和匹配的index targe labels matched_idxs

3. `box_roi_pool`: 通过ROIPool获得各个proposal对应的 feature  (num_proposal, feature_channel, 7, 7)
4. `box_head`: flatten 后进行两层全连接
5. `box_predictor`: 对应分类和bounding box回归的全连接
6. 

### ROIPool

### TwoMLPHead

### FastRCNNPredictor

### fastrcnn_loss
1. 分类损失 
2. 回归损失


