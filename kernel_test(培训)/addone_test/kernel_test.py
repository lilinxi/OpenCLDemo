

#参考：https://tensorflow.google.cn/api_docs/python/tf/nest   tensorflow自带的test类
#参考：Python自带的个单元测试框架unittest：https://zhuanlan.zhihu.com/p/51095152

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
import HTMLTestRunner
import unittest

tf.compat.v1.disable_eager_execution()
sess= tf.compat.v1.Session()
	
#设置为自己路径
nk1 = tf.load_op_library('../addone_cuda/addone_cuda.so') 
nk2 = tf.load_op_library('../addone_opencl/addone_opencl.so')



###############自行设计测试数据#################
#test01 
result01_cuda = sess.run(nk1.add_one([5, 4, 3, 2, 1]))
result01_opencl = sess.run(nk2.nk_add_one([5, 4, 3, 2, 1]))

#test02  
data02 = tf.range(start=1, limit=15, delta=3, dtype=tf.int32)  #数据类型和改写时候输入tensor的类型保持一致
result02_cuda = sess.run(nk1.add_one(data02))
result02_opencl = sess.run(nk2.nk_add_one(data02))


#创建单元测试类，继承自tf.test.TestCase
class AddOneTest(tf.test.TestCase):


	def setUp(self):

		print("\n测试开始:")

	#测试用例01：测试是否能够正常load .so文件（测试用例必须要test开头，否则无法识别）
	def test01(self):

		#打印测试说明
		print ("当取函数边界值1时，验证kernel是否可靠")  
		
		#打印时间
		print ("cuda time use:64.328us  event record time:4.74993us"+"\n"+"opencl time use:23.849us  opencl kernel time use:4.3657us")  

		self.assertAllEqual(result01_cuda, result01_opencl)  #相关判定函数请根据需要查询链接文档（如上），用来测验结果

	
	def test02(self):

		print ("当取函数边界值10000时，验证kernel是否可靠")

		print ("cuda time use:21.288us  event record time:8.39833us"+"\n"+"opencl time use:24.297us  opencl kernel time use:4.47418us")  
		
		self.assertAllEqual(result02_cuda, result02_opencl)

	def tearDown(self):

		print("测试结束!")


if __name__ == '__main__':

    #打印测试结果
	#tf.test.main()

	#--------------------- 生成测试报告--------------

	# 构造测试集
	suite = unittest.TestSuite()
	suite.addTest(AddOneTest("test01"))
	suite.addTest(AddOneTest("test02"))

	now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
	fp = open(now +'-'+ 'result.html', 'wb')

    # 定义报告格式
	runner = HTMLTestRunner.HTMLTestRunner(
		stream=fp,
		title='AddOneKernel测试报告',
		description=u'用例执行情况:')

    # 运行测试用例
	runner.run(suite)
    # 关闭报告文件
	fp.close()


