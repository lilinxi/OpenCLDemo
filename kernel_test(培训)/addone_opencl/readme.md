环境：Tensorflow2.2 CUDA10.2 Python3.7.6

一、改写kernel进行初步测试，看改写后时候正确：

针对于改写的OpenCl Kernel，编写完上述.cc文件后，通过以下操作进行测试：

(1)  linux  shell下直接执行：

TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )

TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

#   XXX.cc为上述自己编写的.cc文件名，XXX.so可自定义，和前边一致即可
g++ -std=c++11 -shared XXX.cc -o XXX.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2  -lOpenCL  


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

g++ -std=c++11 -shared addone_opencl.cc -o addone_opencl.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2  -lOpenCL  

(2) 在addone_opencl.so存放的路径下，进入Python3环境中：

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
sess= tf.compat.v1.Session()

nk = tf.load_op_library('./addone_opencl.so')   #此处可通过dir(nk)，看自己的kernel函数是否加载到nk

print(sess.run(nk.nk_add_one([1,2,3])))    #打印结果为[2 3 4]，且也会打印device设备名称和kernel运行时间，例如：Device: Tesla V100S-PCIE-32GB，以确定是不是GPU执行的。

