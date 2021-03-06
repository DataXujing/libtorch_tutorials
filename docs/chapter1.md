## 第一章 开发环境的搭建
------

<div align=center>
<img src="../img/logo.png" /> 
</div>

<!-- <div align=center>
<img src="../img/ch10/p3.jpg" width=300 /> 
</div>
<br> -->

<!-- https://github.com/AllentDan

https://allentdan.github.io/2021/01/18/libtorch%E6%95%99%E7%A8%8B%EF%BC%88%E4%BA%94%EF%BC%89/

https://www.cnblogs.com/allentbky/p/14315048.html


https://blog.csdn.net/juluwangriyue/article/details/108635280

https://zhuanlan.zhihu.com/p/96397421 -->

### 1. visual studio的安装和配置

visual studio版本最好在2015及以上，本文使用的版本是2017。下载链接：<https://docs.microsoft.com/zh-cn/visualstudio/productinfo/vs2017-system-requirements-vs>具体安装过程可以参考<https://www.jianshu.com/p/320aefbc582d>.打开下载链接下载社区版本即可，安装时对于c++程序设计只需安装对应部分，勾选如下

<div align=center>
<img src="../img/ch1/vs_install.png"  /> 
</div>
<br>

关于visual studio的安装网上有太多教程，这里不再赘述。

### 2. win10 下opencv的安装

1.opencv的下载

+ 进入opencv官网<https://opencv.org/releases.html#>选择对应版本下载.
+ 运行下载后的exe文件，即可解压文件，目录下将会出现名为opencv的文件夹.

<div align=center>
<img src="../img/ch1/cv0.png"  /> 
</div>
<br>

2.环境变量的设置

右击“此电脑”，左击“属性”，“高级系统配置”，“环境变量”，编辑名为“Path”的环境变量，新建以下路径后点击确定，重启计算机。

<div align=center>
<img src="../img/ch1/cv1.png"  /> 
</div>
<br>

3.VS2017中配置opencv

+ 创建工程，工程下建一个源文件
+ 因为我是64位机，所以选择Debug x64
+ 右键项目，依次点击“属性”，“VC++目录”，“包含目录”，将下图路径添加进去后点击确定

<div align=center>
<img src="../img/ch1/cv2.png"  /> 
</div>
<br>


+ 同上述步骤，在“库目录”，将下图路径添加进去后点击确定

<div align=center>
<img src="../img/ch1/cv3.png"  /> 
</div>
<br>

+ 点击“VC++目录”下方的“链接器”，“输入”，“附加依赖项”，添加dll文件后点击确定

<div align=center>
<img src="../img/ch1/cv4.png"  /> 
</div>
<br>


4.测试opencv是否安装成功

在新建的项目中输入如下C++代码(cv_demo项目)

```cpp

#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main()
{
    Mat image = imread("X:\\medicon\\medicon_book\\TensorFlow_book\\libtorch-tutorials\\libtorch-tutorials\\docs\\img\\ch1\\Lenna.jpg");  //图片路径
    double scale = 0.9;  //显示的图像大小为原来大小的0.3
    Size dsize = Size(image.cols*scale, image.rows*scale);
    Mat image2 = Mat(dsize, CV_32S);
    resize(image, image2, dsize);
    imshow("Read Image", image2);
    waitKey(0);
    return 0;


```

<div align=center>
<img src="../img/ch1/cv5.png"  /> 
</div>
<br>

安装配置成功！

### 3. win10下 libtorch的安装和配置

1.libtorch的下载

由于历史原因电脑安装的cuda版本是9.0，这里测试cuda9.0下libtorch的安装和配置，下载地址：<https://download.pytorch.org/libtorch/cu90/libtorch-shared-with-deps-1.0.0.zip>

需要注意的是，libtorch的版本应该大于等于pytorch的训练框架的版本，如果下载最新的libtorch，可以在官网下载

<div align=center>
<img src="../img/ch1/libtorch1.png"  /> 
</div>
<br>

2.解压设置环境变量

解压后，设置环境变量，右键我的电脑->属性->高级系统设置->高级中的环境变量->点击系统变量中的Path->添加dll路径：

<div align=center>
<img src="../img/ch1/libtorch2.png"  /> 
</div>
<br>

3. 新建项目配置环境

+ 添加包含目录

<div align=center>
<img src="../img/ch1/libtorch3.png"  /> 
</div>
<br>

这两个头文件路径中常用的头文件分别是：

```
#include "torch/script.h"
```

和

```
#include "torch/torch.h"
```

网上很多的示例代码添加的是第二个头文件，但是一般都没有说这个头文件所在路径，导致程序找不到很多定义，这个问题网上提到的很少，所以在这里特别说明一下。

+ 设置链接库

添加libtorch包含lib的文件夹路径

<div align=center>
<img src="../img/ch1/libtorch4.png"  /> 
</div>
<br>

+ 添加所需要的lib文件

<div align=center>
<img src="../img/ch1/libtorch5.png"  /> 
</div>
<br>


```
# 我们安装最新的CPU Debug版本！
opencv_world452d.lib
caffe2_detectron_ops.lib
pytorch_jni.lib
caffe2_module_test_dynamic.lib
c10d.lib
torch.lib
torch_cpu.lib
fbjni.lib
mkldnn.lib
dnnl.lib
c10.lib
gloo.lib
fbgemm.lib
asmjit.lib
XNNPACK.lib
libprotocd.lib
libprotobufd.lib
pthreadpool.lib
libprotobuf-lited.lib
cpuinfo.lib
clog.lib
```

+ 修改C/C++ --> 常规 --> SDL检查（否）
+ 修改C/C++ --> 语言 --> 符合模式(否)
+ 运行测试代码

```
#include "torch/torch.h"
#include "torch/script.h"
# include <iostream>


int main()
{
    torch::Tensor output = torch::randn({ 3,2 });
    std::cout << output;

    return 0;
}

```

<div align=center>
<img src="../img/ch1/libtorch6.png"  /> 
</div>
<br>

libtorch配置成功！


### 4.ResNet32分类网络 libtorch部署测试

python下训练的模型转torchscript
```
from torchvision.models import resnet34
import torch.nn.functional as F
import torch.nn as nn
import torch
import cv2

#读取一张图片，并转换成[1,3,224,224]的float张量并归一化
image = cv2.imread("Lenna.jpg")
image = cv2.resize(image,(224,224))
input_tensor = torch.tensor(image).permute(2,0,1).unsqueeze(0).float()/225.0

#定义并加载resnet34模型在imagenet预训练的权重
model = resnet34(pretrained=True)
model.eval()
#查看模型预测该付图的结果
output = model(input_tensor)
output = F.softmax(output,1)
print("模型预测结果为第{}类，置信度为{}".format(torch.argmax(output),output.max()))

#生成pt模型，按照官网来即可
model=model.to(torch.device("cpu"))
model.eval()
var=torch.ones((1,3,224,224))
traced_script_module = torch.jit.trace(model, var)
traced_script_module.save("resnet34.pt")

```

python下的识别结果

<div align=center>
<img src="../img/ch1/libtorch7.png"  /> 
</div>
<br>


```
#include<opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h> 

int main()
{
    //定义使用cuda
    //auto device = torch::Device(torch::kCUDA, 0);
    //读取图片
    auto image = cv::imread("D:\\libtorch_install\\libtorch_code\\resnet32_demo\\python\\lenna.jpg");
    //缩放至指定大小
    cv::resize(image, image, cv::Size(224, 224));
    //转成张量
    auto input_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }).unsqueeze(0).to(torch::kFloat32) / 225.0;
    //加载模型
    auto model = torch::jit::load("D:\\libtorch_install\\libtorch_code\\resnet32_demo\\python\\resnet34.pt");
    //model.to(device);
    model.eval();
    //前向传播
    //auto output = model.forward({ input_tensor.to(device) }).toTensor();
    auto output = model.forward({ input_tensor }).toTensor();
    output = torch::softmax(output, 1);
    std::cout << "模型预测结果为第" << torch::argmax(output) << "类，置信度为" << output.max() << std::endl;
    return 0;
}
```

<div align=center>
<img src="../img/ch1/libtorch8.png"  /> 
</div>
<br>

OK，模型识别结果保持一致，成功！