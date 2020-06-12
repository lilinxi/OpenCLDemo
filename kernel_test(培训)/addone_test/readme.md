经过上述各自cuda kernel和opencl kernel在Python命令行简单确认后，需要自己设计测试用例数据，对改写的kernel进行全方位测试：

(1) 配置环境：在Python3下，查看Python安装路径

import sys
sys.path

通过打印结果找到自己安装路径，此时，把附件中HTMLTestRunner.py文件复制到自己所在Python3安装路径下的site-packages
例如：把HTMLTestRunner.py复制到/home/chengdaguo/anaconda3/lib/python3.7/site-packages下，即Python3安装路径下

(2) 通过kernel_test.py开始对改写的kernel进行全方位测试，测试用例自己根据自己改写的kernel特点，在kernel_test.py中自行设置


最终结果，通过直接运行kernel_test.py，自动生成测试文档，运行后会在当前路径生成一个.html文件，里面即为测试结果

最终提交内容需要包括此测试文件