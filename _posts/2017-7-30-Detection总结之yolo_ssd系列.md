&emsp;&emsp;接着上篇[Detection总结一之RCNN系列](), 这篇讲另一种不同思路做detection的方法体系, 主要介绍三篇论文YOLO, SSD以及YOLO 9000. 在正式介绍以前, 先回归一下RCNN这一基于region proposal的检测体系的基本思路.   

&emsp;&emsp;该类方法首先会先从图像提取出可能存在物体的区域即region proposal; 然后再对这些proposal提取特征; 紧接着用分类器判读该区域是否为某个物体类别, 并利用回归器对该区域的位置进行修正; 最后是一些后处理步骤. 可以看到该方法在语义上可以理解为**"看了"两次图像**.      

## YOLO --- 暴力划分, 只"看"一次图像 
&emsp;&emsp;该方法在名字上很有意思, YOLO(You Only Look Once), 只看一次图像, 从图像输入到输出只有一个网络, 没有额外的proposal步骤. 下面我们来看看其是怎样去掉proposal步骤的. 

**YOLO基本思路步骤**    

![YOLO](https://github.com/JiangQH/jiangqh.github.io/blob/master/images/0730/yolo.png )  

- A. 将图像暴力划分为\\(s \times s\\)的格子区域, 若某个物体的中心落在该格子里, 则由其负责该物体的检测. 每个格子会输出一个表明格子里的物体为某个类别的条件概率. 即\\(P(c_i\|object)\\), 其中\\(c_i = 0, 1, ... C\\).  
- B. 同时假设每个格子可以预测B个物体框(bounding box). 每个物体框有5个输出, 分别是其置信度, 以及对应的坐标x, y, w, h. 其中置信度表达为\\(Pr(object) \times IOU^{truth}_{pred}\\), 其中\\(Pr(object)\\)的取值当有物体时为1否则为0.  
- C. 直接对该输出都使用欧拉距离来进行优化.   

&emsp;&emsp;首先上面格子区域和物体框的设计相当巧妙, YOLO假设每个格子只能有一个类别, 在测试的时候要得到该格子对应的某个类别的概率只需要将A, B中的输出相乘即可. 即 

$$Pr(c_i|object) * Pr(object) * IOU^{truth}_{pred} = Pr(c_i) * IOU^{truth}_{pred}$$ 

&emsp;&emsp;该输出既表达了每个类别的概率, 也表达了预测的框与真实框的重叠度. 

![yolo2](https://github.com/JiangQH/jiangqh.github.io/blob/master/images/0730/yolo2.png)  

&emsp;&emsp;回到网络设计上来. 如上所示YOLO接收一张图像, 经过一系列的卷积层, 然后经过两个全连接层得到最后的输出.    
&emsp;&emsp;由A, B的讨论我们知道每个格子的输出数目应该是\\(C + 5 \times B\\), 同时一共有\\(s \times s\\)个格子. 如上图, 假设\\(s=7, B=2, C=20\\), 即将图分为\\(7 \times 7\\)格子区域, 每个格子输出2个box, 然后一共有20个类别. 那么便可计算最后的输出为\\(7 \times 7 \times 30\\). 注意此处\\(7 \times 7\\)的输出其实是为了方便显示才这样画, 其实是一个全连接层. 同时该\\(7 \times 7\\)输出中的每个30维向量都对应于原图中对应位置的格子区域, 在优化时便会利用该区域的ground truth来进行相应的求loss.  这个30维向量大概长这个样. 

![yolo3](https://github.com/JiangQH/jiangqh.github.io/blob/master/images/0730/yolo3.jpg)    

&emsp;&emsp;最后提一下YOLO中的loss设计, 其loss全部使用平方误差最小化. 为了平衡有无物体之间loss的区别, 加入了两个参数\\(\lambda_{coord}, lambda_{noobj}\\). 同时为了平衡大小框之间的平方误差关系, 实际使用的是开根号后的距离, 这样设计的一个目的是使得较大的框的变化没有较小的框那么重要. 

**YOLO的优缺点**    
&emsp;&emsp;说完了YOLO的基本结构来说说其优缺点. 可以看到YOLO整个网络结构相当粗暴, 没有额外的步骤所以其速度较快, 同时由于其最后所有的bounding box都直接利用了整张feature map来进行预测, 对全局信息做了一个总结, 所以其不易误判, 泛化能力较好.       
&emsp;&emsp;但同时也是因为其直接利用整张feature map来进行所有bounding box的预测, 没有任何先验信息, 使得模型学习困难, 导致最终bounding box的localization error比较大和recall较低. 同时由于模型使用的是最后一个较小的feature, 使得很多较小的物体在该feature上根本就没有了对应的特征, 造成的结果就是对小物体的检测效果比较差. 另外还有一个缺点就是, 该方法对物体的空间分布做了很强的假设, 将图像分为了多个grid, 每个grid仅能有两个box, 且它们只能属于一个类别, 此种假设对于群体物体的检测显然是不合适的.     

## SSD --- 结合RPN和yolo思想, 只做一次forward   
&emsp;&emsp;前面说过 Faster RCNN中的RPN思想相当经典, 后面很多方法都借鉴了其思想. 这里SD以及后面的yolo 9000也不例外. 在介绍SSD以前先来回顾一下RPN的基本思想.   

![rpn](https://github.com/JiangQH/jiangqh.github.io/blob/master/images/0730/rpn.png)

&emsp;&emsp;图像在经过一系列的卷积层操作后得到了一张feature map, 此时在这张feature map上以\\(n \times n\\)大小为单位进行卷积操作, 将每个\\(n \times n\\)的区域中心映射回原rgb图中, 提取不同大小比例固定的k个anchor, 现在该\\(n \times n\\)的feature区域便负责这k个anchor box了. 将该区域的feature经过一定的变换, 最后直接进行是否为box以及对应的坐标修正预测即可.  
&emsp;&emsp;RPN该结构的一个核心便是每一块\\(n \times n\\)的小feature区域作为一个独立的feature单元, 负责k个大小尺度的box的预测和修正. 该种思路的一个基本前提就是feature map上的位置与rgb图上的位置是相对应的.   
&emsp;&emsp;注意上面RPN网络结构中, 分类器仅预测了是否为box, 是个二分类问题.即k个box中每个box对应的输出仅为4+2.    

**SSD基本思路**    

![ssd](https://github.com/JiangQH/jiangqh.github.io/blob/master/images/0730/ssd.PNG)  

&emsp;&emsp;SSD便是借鉴了RPN中的这种bounding box思想. 具体来说, SSD中使用RPN类似的方法结构, 在feature map上也用\\(n \times n\\)的窗口提取feature, 每个\\(n \times n\\)的窗口feature也负责k个default box的判定和修正. 不同于RPN中每个box的输出只判断是否为bounding box的是, 这里需要判断该box的类别, 即如果一共有C类, 那么每个box需要输出用于回归的4个值, 用于分类的C+1(一个背景类)个值. 那么可以得到, 如果一张feature map大小为\\(m \times m\\), 每个\\(n \times n\\)的区域输出k个box, 那么该feature map将一共输出\\(m \times m \times k \times (4 + C + 1)\\).    
&emsp;&emsp;上面\\(n \times n\\)的区域提取可以很自然的用卷积来实现, 跟RPN一样, SSD也成了一个全卷积形式(FCN)的网络结构.   
&emsp;&emsp;除了利用RPN的思想直接预测bounding box外, SSD为了达到在多feature尺度进行预测的效果, 在多个feature map上进行了提取特征训练的方式. 如下图所示.

![ssd2](https://github.com/JiangQH/jiangqh.github.io/blob/master/images/0730/ssd2.PNG)   

&emsp;&emsp;其具体做法是在原来的VGG网络后又添加了几个卷积层, 它们大小逐渐减小, 然后在这些feature map上进行特征的提取和bounding box的预测. 由于feature map的大小不一致, 必然使用的default anchor box的尺寸和比例也应该不一样, SSD中采取的不同feature map大小scale的计算方法是:  
$$s_k = s_{min} + \frac{s_{max} - s_{min}}{m -1}(k-1), k \epsilon [1,m]$$   
&emsp;&emsp;其中\\(s_{min}=0.2,s_{max}=0.95\\), 即最大最小比例的一个均匀分布. 当然和之前计算方法类似, 其损失函数也由分类和回归两部分组成.   

**SSD优缺点**    
&emsp;&emsp;**优点**: 不同于YOLO中在feature map后使用全连接然后让所有bounding box共享该feature的做法, SSD使用了RPN中的思想, 使得每个位置的bounding box都只和局部的feature map关联起来. 这种关联使用了default的anchor box, 这就使得只需要去学习一个offset即可, 这使得学习变得简单起来, 相应而来的则是localization error的减少. 另外一个比较重要的思路是, SSD利用了多尺度的feature map来进行训练操作, 这种操作可以获得不同scale的语义信息, 使得最后的结果accuracy更高, 在低分辨率的图像上也能比较好的工作.   

![yolo](https://github.com/JiangQH/jiangqh.github.io/blob/master/images/0730/yolo.PNG)  

&emsp;&emsp;**缺点**: 缺点之一是SSD对于小物体的检测效果并不好, 这几乎是肯定的. 首先其多scale的feature map训练是在原有的feature map上再叠加了卷积层进行的---而原有的卷积层语义信息已经相当高维, 小物体的信息几乎已经丢失了, 此时再在上面多尺度对于小物体也无意义了, 这也是SSD训练为什么这么依赖数据增强的原因之一(通过crop可以增加小物体在图中的相对比例, 从而使得feature map上的对应区域得到训练), 缺点之二是相比于YOLO中利用了整图feature的情况, 这里只利用了局部的对应特征, 造成的结果就是会存在着误检的情况.  还有一点就是anchor box的选择上还是手动选择, 而手动选择的anchor box与对应的feature map的可视域并不一定贴合.  

## YOLO 9000 --- 结合RPN&联合训练  
&emsp;&emsp;YOLO 9000可以分为两部分, 一部分是YOLO的一个改进版YOLO V2, 说是改进版但其实基本全变掉了, 而我认为其中最重要的改变还是借鉴了RPN的思想, 利用了anchor box来进行预测.  另一部分则是提出了一种联合训练机制, 使得能够进行detection的种类大大增加.    

**YOLO V2的基本改进**   
- 1、加入Batch Normalization & 在ImageNet上先使用高分辨率的图来进行finetune.   
- 2、加入了来自前面层的跳过连接, 得到更细节的feature.  
- 3、完全抛弃了原来的训练方式, 而是使用RPN中类似的anchor box思路来进行bounding box的预测.    
- 4、由于现在是FCN的结构了, 所以在训练的时候使用Multi-Scale training, 即在训练时随机变换图像大小, 带来更大的鲁棒性.    
- 5、重新训练了一个网络结构Darknet 19

&emsp;&emsp;下面来仔细说说前面几点, 第一点不用多说. 第三点加跳过连接其实是一个很常用的思路, 目的是为了获得一些更细节的feature, 同时对于小物体来说, 加了跳过连接使得小物体的feature不会丢失. 第四点也不必多说, 训练时的图像大小随机变化必然能带来鲁棒性的增加. 关于第三点使用RPN类似结构我们来仔细看看.   
&emsp;&emsp;首先和RPN以及SSD类似, 每块\\(n \times n\\)feature负责k个anchor box, 不过这里每个anchor box的输出延续了YOLO原来的思路, 但将class的概率解耦出来, 为每个anchor box预测概率. 即每个anchor box对应的输出为class的概率, object的概率(和真实物体的IOU), 和坐标值. 即   

![yolo2](https://github.com/JiangQH/jiangqh.github.io/blob/master/images/0730/yolo2.jpg)   

&emsp;&emsp;同时YOLO V2假设图像中较大的物体基本就在图像中央, 所以想要只有一个对应的feature区域来负责该物体, 那么最后的feature map大小便应该是奇数---所以调节了网络结构, 使得feature map的输出为奇数.  Anchor box的使用使得原来直接预测位置变为了预测offset, 使得模型的localization error减低了很多.  
&emsp;&emsp;除了使用anchor box的思路外, YOLO V2还在设计上**做了两个改进**. 第一个是原来每个\\(n \times n\\)的区域负责的k个anchor box的比例以及scale是手动选择的, YOLO V2对训练集真实的box利用距离衡量1-IOU(box, center)来进行了k-means聚类, 最终挑选得到了5个default的anchor box. 第二个是改进了坐标的回归方式使得模型更稳定, 在RPN的原始结构中, 对于坐标会进行预测\\(\(t_x, t_y\)\\), 而对应的\\(\(x,y\)\\)的计算为:  
$$x = (t_x \times w_a) + x_a$$   
$$y = (t_y \times h_a) + y_a$$  
&emsp;&emsp;注意到该公式没有任何约束, \\(t_x\\)每变化一个单位对应的坐标便会左移或者右移一个\\(w_a\\)的单位, 所以理论上任何一个anchor box都可以出现在图中的任何位置. 这显然会带来模型的不稳定性, 使得训练变得困难. 我们所希望的是anchor box只在其default区域的一定范围内移动. 为了解决这个问题YOLO V2使用了YOLO中的预测相对位置的办法 --- 预测某个anchor box的中心点相对于其对应的grid cell的左上角的位置. 同时使用logistic 激活函数使得网络的输出在0-1之间. 那么此时为每个anchor box的预测输出, 除了类别概率外便可如下计算  

![yolo2_rpn](https://github.com/JiangQH/jiangqh.github.io/blob/master/images/0730/yolo2_rpn.png)  

**YOLO 9000的联合训练**     
&emsp;&emsp;标定检测数据集的标定类别数量其实相对较少, 但是分类数据集的标定类别却很大. 而且检测数据集往往细分程度不够, 例如只会细分到"猫", "狗"等, 而分类数据集则细分程度很大, 可以细分到"加菲猫", "哈士奇"等等. 作者用了一种层级的结构将两种数据集结合起来, 然后将数据混合训练, 便能在检测时具体检测出"加菲猫", "哈士奇"等细分物体了.  而数据结合的关键便是构建层级树, 使得检测集中的某个物体类别下能分支出很多个类别来, 最后的效果便是一个world tree.   

![world_tree](https://github.com/JiangQH/jiangqh.github.io/blob/master/images/0730/world_tree.png)  

&emsp;&emsp;在进行训练的时候, 如果遇到某张图像是来自检测数据集, 则计算整个loss函数然后进行反向传播和更新, 如果遇到来自分类数据集则只对分类部分的loss进行计算和反向传播更新. 需要注意的是此种方法对数据集有一定的要求.  

**YOLO V2的优缺点**    
&emsp;&emsp;YOLO V2由于在anchor box的loss设计上做了改进, 同时使用了跳过连接等方法, 加上大量的训练技巧, 目前来说在实时的物体检测方法中可以说是做得最好的了.    

&emsp;&emsp;但YOLO V2可以说也是和SSD一样也是RPN一脉的方法, 虽然其在损失函数以及anchor box等的设计上做了改进, 但是该类方法的一些共有缺点它还是具有. 首先它们其实都只用了一小部分feature, 虽然说高层的feature具有语义信息, 但结segmentation领域的一些先验知识来看这还是不够的, 造成的结果就是存在着误判. 其次就是它们利用的feature其实都偏于高层, 要想对小物体以及细节信息有更准确的把握, 利用低层feature肯定是有用的, 这在今年的cvpr17的工作FPN中得到了印证, 后面将会介绍.    