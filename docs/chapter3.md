## 第三章 模型搭建
------

### 1.基本模块的搭建

模块化编程的思想非常重要，通过模块化编程可以大幅减少重复的敲代码过程，同时代码可读性也会增加。本章将讲述如何使用libtorch搭建一些MLP和CNN的基本模块。


#### 1.MLP基本单元

首先是线性层的声明和定义，包括初始化和前向传播函数。代码如下：

```cpp
// LinearBnReluImpl 类
class LinearBnReluImpl : public torch::nn::Module{
    public:
        LinearBnReluImpl(int intput_features, int output_features);
        torch::Tensor forward(torch::Tensor x);
    private:
        //layers
        torch::nn::Linear ln{nullptr};  //ln是指向类，结构或联合的指针
        torch::nn::BatchNorm1d bn{nullptr};
};
TORCH_MODULE(LinearBnRelu);

// 类的构造函数
LinearBnReluImpl::LinearBnReluImpl(int in_features, int out_features){
    ln = register_module("ln", torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features)));
    bn = register_module("bn", torch::nn::BatchNorm1d(out_features));
}

// 类的forword成员函数
torch::Tensor LinearBnReluImpl::forward(torch::Tensor x){
    x = torch::relu(ln->forward(x));  
    x = bn(x);
    return x;
}
```

在MLP的构造线性层模块类时，我们继承了`torch::nn::Module`类，将初始化和前向传播模块作为public，可以给对象使用，而里面的线性层`torch::nn::Linear`和归一化层`torch::nn::BatchNorm1d`被隐藏作为私有变量。

定义构造函数时，需要将原本的指针对象ln和bn进行赋值，同时将两者的名称也确定。前向传播函数就和pytorch中的forward类似。

#### 2.CNN基本单元

CNN的基本单元构建和MLP的构建类似，但是又稍有不同，首先需要定义的时卷积超参数确定函数。

```cpp
inline torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
    int64_t stride = 1, int64_t padding = 0, bool with_bias = false) {
        torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
        conv_options.stride(stride);
        conv_options.padding(padding);
        conv_options.bias(with_bias);
        return conv_options;
}
```

该函数返回`torch::nn::Conv2dOptions`对象，对象的超参数由函数接口指定，这样可以方便使用。同时指定inline(内联函数），提高Release模式下代码执行效率。

```
C++ 内联函数是通常与类一起使用。如果一个函数是内联的，那么在编译时，编译器会把该函数的代码副本放置在每个调用该函数的地方。

对内联函数进行任何修改，都需要重新编译函数的所有客户端，因为编译器需要重新更换一次所有的代码，否则将会继续使用旧的函数。

如果想把一个函数定义为内联函数，则需要在函数名前面放置关键字 inline，在调用函数之前需要对函数进行定义。如果已定义的函数多于一行，编译器会忽略 inline 限定符。
```

随后则是和MLP的线性模块类似，CNN的基本模块由卷积层，激活函数和归一化层组成。代码如下：

```cpp
class ConvReluBnImpl : public torch::nn::Module {
    public:
        ConvReluBnImpl(int input_channel=3, int output_channel=64, int kernel_size = 3, int stride = 1); // 构造函数
        torch::Tensor forward(torch::Tensor x);  //forward成员函数
    private:
        // Declare layers
        torch::nn::Conv2d conv{ nullptr };
        torch::nn::BatchNorm2d bn{ nullptr };
};
TORCH_MODULE(ConvReluBn);

// 构造函数的实现
ConvReluBnImpl::ConvReluBnImpl(int input_channel, int output_channel, int kernel_size, int stride) {
    conv = register_module("conv", torch::nn::Conv2d(conv_options(input_channel,output_channel,kernel_size,stride,kernel_size/2)));
    bn = register_module("bn", torch::nn::BatchNorm2d(output_channel));

}

// forward函数的实现
torch::Tensor ConvReluBnImpl::forward(torch::Tensor x) {
    x = torch::relu(conv->forward(x));
    x = bn(x);
    return x;
}

```
每一层的实现均是通过前面定义的基本模块LinearBnRelu。



### 2.简单的MLP

在MLP的例子中，我们以搭建一个四层感知机为例，介绍如何使用cpp实现深度学习模型。该感知机接受in_features个特征，输出out_features个编码后的特征。中间特征数定义为32，64和128。

```cpp
class MLP: public torch::nn::Module{  // 继承
    public:
        MLP(int in_features, int out_features);
        torch::Tensor forward(torch::Tensor x);
    private:
        int mid_features[3] = {32,64,128};
        LinearBnRelu ln1{nullptr};
        LinearBnRelu ln2{nullptr};
        LinearBnRelu ln3{nullptr};
        torch::nn::Linear out_ln{nullptr};
};

// 构造函数的实现
MLP::MLP(int in_features, int out_features){
    ln1 = LinearBnRelu(in_features, mid_features[0]);
    ln2 = LinearBnRelu(mid_features[0], mid_features[1]);
    ln3 = LinearBnRelu(mid_features[1], mid_features[2]);
    out_ln = torch::nn::Linear(mid_features[2], out_features);

    ln1 = register_module("ln1", ln1);
    ln2 = register_module("ln2", ln2);
    ln3 = register_module("ln3", ln3);
    out_ln = register_module("out_ln",out_ln);
}

// forward函数的实现
torch::Tensor MLP::forward(torch::Tensor x){
    x = ln1->forward(x);  //指针成员的访问调用
    x = ln2->forward(x);
    x = ln3->forward(x);
    x = out_ln->forward(x);
    return x;
}
```
每一层的实现均是通过前面定义的基本模块LinearBnRelu。

### 3.简单CNN

前面介绍了构建CNN的基本模块ConvReluBn，接下来尝试用c++搭建CNN模型。该CNN由三个stage组成，每个stage又由一个卷积层一个下采样层组成。这样相当于对原始输入图像进行了8倍下采样。中间层的通道数变化与前面MLP特征数变化相同，均为输入->32->64->128->输出。

```cpp

class plainCNN : public torch::nn::Module{
    public:
        plainCNN(int in_channels, int out_channels);
        torch::Tensor forward(torch::Tensor x);
    private:
        int mid_channels[3] = {32,64,128};
        ConvReluBn conv1{nullptr};
        ConvReluBn down1{nullptr};
        ConvReluBn conv2{nullptr};
        ConvReluBn down2{nullptr};
        ConvReluBn conv3{nullptr};
        ConvReluBn down3{nullptr};
        torch::nn::Conv2d out_conv{nullptr};
};

// 构造函数的实现
plainCNN::plainCNN(int in_channels, int out_channels){
    conv1 = ConvReluBn(in_channels,mid_channels[0],3);
    down1 = ConvReluBn(mid_channels[0],mid_channels[0],3,2);
    conv2 = ConvReluBn(mid_channels[0],mid_channels[1],3);
    down2 = ConvReluBn(mid_channels[1],mid_channels[1],3,2);
    conv3 = ConvReluBn(mid_channels[1],mid_channels[2],3);
    down3 = ConvReluBn(mid_channels[2],mid_channels[2],3,2);
    out_conv = torch::nn::Conv2d(conv_options(mid_channels[2],out_channels,3));

    conv1 = register_module("conv1",conv1);
    down1 = register_module("down1",down1);
    conv2 = register_module("conv2",conv2);
    down2 = register_module("down2",down2);
    conv3 = register_module("conv3",conv3);
    down3 = register_module("down3",down3);
    out_conv = register_module("out_conv",out_conv);
}

// forward 方法的使用
torch::Tensor plainCNN::forward(torch::Tensor x){
    x = conv1->forward(x);
    x = down1->forward(x);
    x = conv2->forward(x);
    x = down2->forward(x);
    x = conv3->forward(x);
    x = down3->forward(x);
    x = out_conv->forward(x);
    return x;
}
```

假定输入一个三通道图片，输出通道数定义为n，输入表示一个`[1,3,224,224]`的张量，将得到一个`[1,n,28,28]`的输出张量。


### 4.简单LSTM

最后则是一个简单的LSTM的例子，用以处理时序型特征。在直接使用`torch::nn::LSTM`类之前，我们先定一个返回`torch::nn::LSTMOptions`对象的函数，该函数接受关于LSTM的超参数，返回这些超参数定义的结果。

```cpp
inline torch::nn::LSTMOptions lstmOption(int in_features, int hidden_layer_size, int num_layers, bool batch_first = false, bool bidirectional = false){
    torch::nn::LSTMOptions lstmOption = torch::nn::LSTMOptions(in_features, hidden_layer_size);
    lstmOption.num_layers(num_layers).batch_first(batch_first).bidirectional(bidirectional);
    return lstmOption;
}

//batch_first: true for io(batch, seq, feature) else io(seq, batch, feature)
class LSTM: public torch::nn::Module{
public:
    LSTM(int in_features, int hidden_layer_size, int out_size, int num_layers, bool batch_first);
    torch::Tensor forward(torch::Tensor x);
private:
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear ln{nullptr};
    std::tuple<torch::Tensor, torch::Tensor> hidden_cell;
};
```

声明好LSTM以后，我们将内部的初始化函数和前向传播函数实现如下：

```cpp
LSTM::LSTM(int in_features, int hidden_layer_size, int out_size, int num_layers, bool batch_first){
    lstm = torch::nn::LSTM(lstmOption(in_features, hidden_layer_size, num_layers, batch_first));
    ln = torch::nn::Linear(hidden_layer_size, out_size);

    lstm = register_module("lstm",lstm);
    ln = register_module("ln",ln);
}

torch::Tensor LSTM::forward(torch::Tensor x){
    auto lstm_out = lstm->forward(x);
    auto predictions = ln->forward(std::get<0>(lstm_out));
    return predictions.select(1,-1);
}

```

感谢大佬开源： [libtorch教程（三）](https://allentdan.github.io/2021/01/16/libtorch%E6%95%99%E7%A8%8B%EF%BC%88%E4%B8%89%EF%BC%89/)