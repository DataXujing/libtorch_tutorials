## 第九章 总结和展望
------

首选要说明的是该教程是对 Allent Dan大佬开源的libtorch教程的一个整理，其开源了大量的libtorch在C++下训练和部署的方法。其GitHub地址为：<https://github.com/AllentDan>
建议大家看并给Star：

+ https://github.com/AllentDan/LibtorchTutorials
+ https://github.com/AllentDan/LibtorchDetection
+ https://github.com/AllentDan/LibtorchSegmentation


现在深度学习相关的岗位，尤其是CV岗，python的HC真的不多，大厂还好，有但是要求高，小厂要求低些但是也基本要求会部署之类的。一般大厂研究院可以搞训练和部署分开搞，中小厂一般还是c++和python都有要求的，甚至很多大厂也是同时要求训练部署一个人包。所以，做深度学习相关的，对c++有些要求，或者说会部署调优确实很重要了。

Allent Dan大佬关于libtorch的一些编程经验，总结下吧：

libtorch有许多坑，比如sequential in sequential会报错，而且解决不了，除非另写个stack的sequential类….比如CPU比python有时候还要慢一些…虽然GPU一般会快个三成。

libtorch中的sequential，不能堆叠std::vector\torch::Tensor\为输入的模块，比如yolov5模型，就整了个ConCat模块，libtorch里面没法用啊。一时间不知道该怪yolov5作者代码规范差好，还是怪libtorch垃圾好，还是怪自己没资源好。所以最后整了个yolov4_tiny。

libtorch还是有许多待优化的东西的，比如提速啥的，后面要是能整个int8精度预测，一个api调用，可能TensorRT的市场份额又得缩小了。当然，应该没有TensorRT提速那么多，毕竟不可能比人家更了解Nvidia显卡了…

有条件，还是自己整个配套的python模型，然后自己训练个预训练的权重，比到处找合适的开源项目及权重香多了。当然，要是真的精力好，自己从头训练libtorch的模型也未尝不可啊筒子们。或许等我项目完善起来，也可以实现libtorch从头训练吧，需要加许多东西，加许多数据增强，一些学习率调整策略也要自己实现，加上时间和资源，应该还是可以的。毕竟libtorch接过了caffe的大旗，而且有脸书大厂在维护。

说到底，就是资源给够，花钱铺人，写个libtorch++都可以实现…想必许多公司应该就是这样干的，目前也在愉快融资啥的了。

整体来说如果你是用Pytorch的模型在C++部署时选择的方案只能是libtorch或TensorRT,而对于libtorch， Facebook已经提供了很好的预编译的C++库给我们，不像TensorFlow，你需要自己编译C++的库，这样很方便在C++中使用。





