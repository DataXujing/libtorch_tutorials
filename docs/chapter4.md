## 第四章 数据加载模块
------

本章将详细介绍如何使用libtorch自带的数据加载模块，使用该模块是实现模型训练的重要条件。除非这个数据加载模块功能不够，不然继承libtorch的数据加载类还是很有必要的，简单高效。

### 1.使用前置条件

libtorch提供了丰富的基类供用户自定义派生类，`torch::data::Dataset`就是其中一个常用基类。使用该类需要明白基类和派生类，以及所谓的继承和多态。有c++编程经验者应该都不会陌生，为方便不同阶段读者就简单解释一下吧。类就是父亲，可以生出不同的儿子，生儿子叫派生或者继承(看使用语境)，生不同的儿子就实现了多态。父亲就是基类，儿子就是派生类。现实中，父亲会把自身的一部分财产留下来养老，儿子们都不能碰，这就是private了，部分财产儿子能用，但是儿子的对象不能用，这叫protected，还有些财产谁都能用就是public。和现实中的父子类似，代码中，派生类可以使用父类的部分属性或者函数，全看父类怎样定义。

然后理解一下虚函数，就是父亲指定了部分财产是public的，但是是用来买房的，不同的儿子可以买不同的房子，可以全款可以贷款，这就是财产在父亲那就是virtual的。子类要继承这个virtual财产可以自己重新规划使用方式。

事实上，如果有过pytorch的编程经验者很快会发现，libtorch的Dataset类的使用和python下使用非常相像。pytorch自定义dataload，需要定义好Dataset的派生类，包括初始化函数`init`，获取函数`getitem`以及数据集大小函数`len`。类似的，libtorch中同样需要处理好初始化函数，`get()`函数和`size()`函数。


### 2.图片文件遍历

下面以分类任务为例，介绍libtorch的Dataset类的使用。使用pytorch官网提供的[昆虫分类数据集](https://download.pytorch.org/tutorial/hymenoptera_data.zip)，下载到本地解压。将该数据集根目录作为索引，实现Dataloader对图片的加载。

首先定义一个加载图片的函数，使用网上出现较多的c++遍历文件夹的代码，将代码稍作修改如下：

```cpp
//遍历该目录下的.jpg图片
//函数声明
// https://github.com/AllentDan/LibtorchTutorials/tree/main/lesson4-DatasetUtilization
void load_data_from_folder(std::string image_dir, std::string type, std::vector<std::string> &list_images, std::vector<int> &list_labels, int label);

void load_data_from_folder(std::string path, std::string type, std::vector<std::string> &list_images, std::vector<int> &list_labels, int label)
{
    // 声明变量
    long long hFile = 0; //句柄
    struct _finddata_t fileInfo;
    std::string pathName;

    if ((hFile = _findfirst(pathName.assign(path).append("\\*.*").c_str(), &fileInfo)) == -1)
    {
        return;
    }
    do
    {
        const char* s = fileInfo.name;
        const char* t = type.data();

        if (fileInfo.attrib&_A_SUBDIR) //是子文件夹
        {
            //遍历子文件夹中的文件(夹)
            if (strcmp(s, ".") == 0 || strcmp(s, "..") == 0) //子文件夹目录是.或者..
                continue;
            std::string sub_path = path + "\\" + fileInfo.name;
            label++;
            load_data_from_folder(sub_path, type, list_images, list_labels, label);

        }
        else //判断是不是后缀为type文件
        {
            if (strstr(s, t))
            {
                std::string image_path = path + "\\" + fileInfo.name;
                list_images.push_back(image_path);
                list_labels.push_back(label);
            }
        }
    } while (_findnext(hFile, &fileInfo) == 0);
    return;
}
```

修改后的函数接受数据集文件夹路径image_dir和图片类型image_type，将遍历到的图片路径和其类别分别存储到list_images和list_labels，最后lable变量用于表示类别计数。传入lable=-1，返回的lable值加一后等于图片类别。


### 3.自定义Dataset

定义dataSetClc，该类继承自`torch::data::Dataset`。定义私有变量image_paths和labels分别存储图片路径和类别，是两个vector变量。dataSetClc的初始化函数就是加载图片和类别。通过get()函数返回由图像和类别构成的张量列表。可以在get()函数中做任意针对图像的操作，如数据增强等。效果等价于pytorch中的getitem中的数据增强。

```cpp
class dataSetClc:public torch::data::Dataset<dataSetClc>{
public:
    int class_index = 0;
    dataSetClc(std::string image_dir, std::string type){
        load_data_from_folder(image_dir, std::string(type), image_paths, labels, class_index-1);
    }
    // Override get() function to return tensor at location index
    // 重写get()和size()方法
    torch::data::Example<> get(size_t index) override{
        std::string image_path = image_paths.at(index);  //vector的切片
        cv::Mat image = cv::imread(image_path);
        cv::resize(image, image, cv::Size(224, 224)); //尺寸统一，用于张量stack，否则不能使用stack
        int label = labels.at(index);
        torch::Tensor img_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width
        torch::Tensor label_tensor = torch::full({ 1 }, label);
        return {img_tensor.clone(), label_tensor.clone()};
    }
    // Override size() function, return the length of data
    torch::optional<size_t> size() const override {
        return image_paths.size();
    };
private:
    std::vector<std::string> image_paths;
    std::vector<int> labels;
};
```


### 4.使用自定义的Dataset

下面使用定义好的数据加载类，以昆虫分类中的训练集作为测试，代码如下。可以打印加载的图片张量和类别。

```cpp
int batch_size = 2;
std::string image_dir = "your path to\\hymenoptera_data\\train";
auto mdataset = myDataset(image_dir,".jpg").map(torch::data::transforms::Stack<>());
auto mdataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(mdataset), batch_size);
for(auto &batch: *mdataloader){
    auto data = batch.data;
    auto target = batch.target;
    std::cout<<data.sizes()<<target;
}
```