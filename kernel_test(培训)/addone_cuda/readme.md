环境：Tensorflow2.2 CUDA10.2 Python3.7.6

addone cuda kernel源码路径：\tensorflow\examples\adding_an_op\


一、改写kernel进行初步测试，看改写后时候正确：

针对于cuda Kernel，直接拿出.cc和.cu.cc (有的还有.h文件)，通过以下操作进行测试：

(1)  linux  shell下直接执行：

TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )

TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

#   XXX.cu.cc为已知的.cu.cc文件名，XXX.cu.o可自定义，和后边一致即可
nvcc -std=c++11 -c -o XXX.cu.o XXX.cu.cc   ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr

#   XXX.cc为上述已知的.cc文件名，XXX.cu.o为上个命令生成的文件，XXX.so 可自定义，和前边名字一致即可
g++ -std=c++11 -shared -o XXX.so  XXX.cc  XXX.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}


注：问题解决：若在执行nvcc编译命令时，报错：
/home/dell01/.local/lib/python3.6/site-packages/tensorflow/include/tensorflow/core/util/gpu_kernel_helper.h:22:10: fatal error: third_party/gpus/cuda/include/cuda_fp16.h: 没有那个文件或目录
 #include "third_party/gpus/cuda/include/cuda_fp16.h"

解决办法：找到tensorflow安装路径下的third_party，例如：/home/chengdaguo/anaconda3/lib/python3.7/site-packages/tensorflow/include/third_party
在下面：
'''
mkdir gpus && cd gpus
mkdir cuda && cd cuda 
cp cuda10.2的安装目录中所有内容 到cuda下 //例如：cp -r /usr/local/cuda10.2/* /home/chengdaguo/anaconda3/lib/python3.7/site-packages/tensorflow/include/third_party/gpus/cuda/

'''


(2) 在XXX.so存放的路径下，进入Python环境中：

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
sess= tf.compat.v1.Session()

nk = tf.load_op_library('./XXX.so')   #此处可通过dir(nk)，看自己的kernel函数是否加载到nk下，此处统一用nk

#   调用opencl  kernel执行操作，并打印结果，name为REGISTER_OP("name") 中的名字，但是注意由于C++下命名规则是首字母大写格式，例如：NkAddOne
#   这里通过封装后转换成了Python接口，它会自动转换为Python命名格式，首字母小写并加下划线，即类似这样：nk_add_one ，data为输入数据

print(sess.run(t.name(data))) 


AddOne上述过程示例：
(1)linux  shell下直接执行：

TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )

TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 -c -o addone_cuda.cu.o addone_cuda.cu.cc   ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr

g++ -std=c++11 -shared -o addone_cuda.so addone_cuda.cc   addone_cuda.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}

(2) 在addone_cuda.so存放的路径下，进入Python3环境中：

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
sess= tf.compat.v1.Session()

nk = tf.load_op_library('./addone_cuda.so')   #此处可通过dir(nk)，看自己的kernel函数是否加载到nk

print(sess.run(nk.add_one([1,2,3])))    #打印结果为[2 3 4]

