## 第二章 张量的常规操作
------

<!-- <div align=center>
<img src="../img/logo.png" /> 
</div> -->

ibtorch(pytorch c++)的大多数api和pytorch保持一致，因此，libtorch中张量的初始化也和pytorch中的类似。本文介绍四种深度图像编程需要的初始化方法。

### 1.Tensor的初始化

第一种，固定尺寸和值的初始化。

```
//常见固定值的初始化方式
auto b = torch::zeros({3,4});
b = torch::ones({3,4});
b= torch::eye(4);
b = torch::full({3,4},10);
b = torch::tensor({33,22,11});

```
pytorch中用`[]`表示尺寸，而cpp中用`{}`表示。zeros产生值全为0的张量。ones产生值全为1的张量。eye产生单位矩阵张量。full产生指定值和尺寸的张量。`torch::tensor({})`也可以产生张量，效果和pytorch的`torch.Tensor([])`或者`torch.tensor([])`一样。

第二种，固定尺寸，随机值的初始化方法

```
//随机初始化
auto r = torch::rand({3,4});
r = torch::randn({3, 4});
r = torch::randint(0, 4,{3,3});
```

rand产生0-1之间的随机值，randn取正态分布N(0,1)的随机值，randint取`[min,max)`的随机整型数值。

第三种，从c++的其他数据类型转换而来

```
int aa[10] = {3,4,6};
std::vector<float> aaaa = {3,4,6};
auto aaaaa = torch::from_blob(aa,{3},torch::kFloat);
auto aaa = torch::from_blob(aaaa.data(),{3},torch::kFloat);
```

pytorch可以接受从其他数据类型如numpy和list的数据转化成张量。libtorch同样可以接受其他数据指针，通过`from_blob`函数即可转换。这个方式在部署中经常用到，如果图像是opencv加载的，那么可以通过`from_blob`将图像指针转成张量。

第四种，根据已有张量初始化

```
auto b = torch::zeros({3,4});
auto d = torch::Tensor(b);
d = torch::zeros_like(b);
d = torch::ones_like(b);
d = torch::rand_like(b,torch::kFloat);
d = b.clone();
```

这里，`auto d = torch::Tensor(b)`等价于`auto d = b`，两者初始化的张量d均受原张量b的影响，b中的值发生改变，d也将发生改变，但是b如果只是张量变形，d却不会跟着变形，仍旧保持初始化时的形状，这种表现称为浅拷贝。zeros_like和ones_like顾名思义将产生和原张量b相同形状的0张量和1张量，randlike同理。最后一个clone函数则是完全拷贝成一个新的张量，原张量b的变化不会影响d，这被称作`深拷贝`。

### 2.Tensor的变形

torch改变张量形状，不改变张量存储的data指针指向的内容，只改变张量的取数方式。libtorch的变形方式和pytorch一致，有view，transpose，reshape，permute等常用变形。

```
auto b = torch::full({10},3);
b.view({1, 2,-1});
std::cout << b;

b = b.view({1, 2,-1});
std::cout << b;

auto c = b.transpose(0,1);
std::cout << c;

auto d = b.reshape({1,1,-1});
std::cout << d;

auto e = b.permute({1,0,2});
std::cout << e;
```

`.view`不是inplace操作，需要加=。变形操作没太多要说的，和pytorch一样。还有squeeze和unsqueeze操作，也与pytorch相同。

### 3.Tensor的切片

通过索引截取张量，代码如下

```
auto b = torch::rand({10,3,28,28});
std::cout << b[0].sizes(); //第0张照片
std::cout << b[0][0].sizes(); //第0张照片的第0个通道
std::cout << b[0][0][0].sizes(); //第0张照片的第0个通道的第0行像素 dim为1
std::cout << b[0][0][0][0].sizes(); //第0张照片的第0个通道的第0行的第0个像素 dim为0
```

除了索引，还有其他操作是常用的，如narrow，select，index，index_select。

```
std::cout << b.index_select(0,torch::tensor({0, 3, 3})).sizes(); //选择第0维的0，3，3组成新张量[3,3,28,28]
std::cout << b.index_select(1,torch::tensor({0,2})).sizes(); //选择第1维的第0和第2的组成新张量[10, 2, 28, 28]
std::cout << b.index_select(2,torch::arange(0,8)).sizes(); //选择十张图片每个通道的前8列的所有像素[10, 3, 8, 28]
std::cout << b.narrow(1,0,2).sizes(); //选择第1维，从0开始，截取长度为2的部分张量[10, 2, 28, 28]
std::cout << b.select(3,2).sizes(); //选择第3维度的第二个张量，即所有图片的第2行组成的张量[10, 3, 28]
```

index需要单独说明用途。在pytorch中，通过掩码Mask对张量进行筛选是容易的直接`Tensor[Mask]`即可。但是c++中无法直接这样使用，需要index函数实现，代码如下：

```
auto c = torch::randn({3,4});
auto mask = torch::zeros({3,4});
mask[0][0] = 1;
std::cout<<c;
std::cout<<c.index({mask.to(torch::kBool)});
```

有网友提问，这样index出来的张量是深拷贝的结果，也就是得到一个新的张量，那么如何对原始张量的mask指向的值做修改呢。查看torch的api发现还有index_put_函数用于直接放置指定的张量或者常数。组合index_put_和index函数可以实现该需求。

```
auto c = torch::randn({ 3,4 });
auto mask = torch::zeros({ 3,4 });
mask[0][0] = 1;
mask[0][2] = 1;
std::cout << c;
std::cout << c.index({ mask.to(torch::kBool) });
std::cout << c.index_put_({ mask.to(torch::kBool) }, c.index({ mask.to(torch::kBool) })+1.5);
std::cout << c;
```

此外python中还有一种常见取数方式`tensor[:,0::4]`这种在第1维，起始位置为0，间隔4取数的方式，在c++中直接用`slice`函数实现。

### 4.Tensor间的操作

拼接和堆叠

```
auto b = torch::ones({3,4});
auto c = torch::zeros({3,4});
auto cat = torch::cat({b,c},1); //1表示第1维，输出张量[3,8]
auto stack = torch::stack({b,c},1); //1表示第1维，输出[3,2,4]
std::cout << b << c << cat << stack;
```

到这读者会发现，从pytorch到libtorch，掌握了`[]`到`{}`的变化就简单很多，大部分操作可以直接迁移。

四则运算操作同理，像对应元素乘除直接用`*`和`/`即可，也可以用`.mul`和`.div`。矩阵乘法用`.mm`，加入批次就是`.bmm`。

```
auto b = torch::rand({3,4});
auto c = torch::rand({3,4});
std::cout << b << c << b*c << b/c << b.mm(c.t());

```

其他一些操作像`clamp`，`min`，`max`这种都和pytorch类似，仿照上述方法可以自行探索。

感谢大佬：[libtorch教程（二）](https://allentdan.github.io/2021/01/16/libtorch%E6%95%99%E7%A8%8B%EF%BC%88%E4%BA%8C%EF%BC%89/)