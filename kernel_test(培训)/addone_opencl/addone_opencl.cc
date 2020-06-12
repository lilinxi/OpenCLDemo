#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

/* 
通过REGISTER_OP 宏来定义Op的接口，定义Op输入输出和属性：
声明改写kernel的名字：必须使用驼峰命名法即所有单词首字母大写（并且不能和原改写kernel名字一样，命名格式为：Nk+原Kernel名字，例如：ClAddOne
输入（名字，类型）、输出（名字，类型）等等一些属性。
以上内容，可在tensorflow源码中找到所改写的CUDA下的kernel源码，找到上述相应内容，借鉴过来即可。
*/
REGISTER_OP("NkAddOne")  //后面调用改写的kernel函数进行测试时，调用的名字即为这个
    //数据类型针对自己改写的kernel进行修改（参考tensorflow下cuda kernel源码,\tensorflow\examples\adding_an_op\cuda_op_kernel.cc中）
    .Input("input: int32")    
    .Output("output: int32");


/*
把改写的OpenCL kernel 封装为Op的具体实现，其中主要内容即为OpenCL编程过程，即编写一个OpenCl代码时的所进行的步骤，和CUDA是无关的，
*/

//其中参数即输入和输出,指针in存放输入，指针res存放输出，在主函数调用时，通过tensorflow下的输入输出张量获取，如251行，此处也注意修改对应数据类型
void ClKernelLauncher(const int* in, const int N, int* res) {    

    //首先获得系统上所有的OpenCL platform，调用两次clGetPlatformIDs函数，第一次获取可用的平台数量，第二次获取一个可用的平台ID,不用动即可。
    cl_int err;
    cl_uint num;
    err = clGetPlatformIDs(0, 0, &num);
    if (err != CL_SUCCESS)
    {
        std::cout << "Unable to get platforms\n";
        return;
    }

    std::vector<cl_platform_id> platforms(num);
    err = clGetPlatformIDs(num, &platforms[0], &num);
    if (err != CL_SUCCESS)
    {
        std::cout << "Unable to get platform ID\n";
        return;
    }

    
    /*
    调用clCreateContextFromType创建一个上下文（context），即OpenCL的Platform上共享和使用资源的环境，包括kernel、device、memory objects、command queue等。
    使用中一般一个Platform对应一个Context。不用动即可。
    */
    cl_context_properties prop[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[0]), 0};
    cl_context context = clCreateContextFromType(prop, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, NULL); 
    if (context == 0)
    {
        std::cout << "Can't create OpenCL context\n";
        return;
    }


   //获得装置列表
    size_t cb;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
    std::vector<cl_device_id> devices(cb / sizeof(cl_device_id));
    clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &cb);

    //获取装置名称
    std::string devname;
    devname.resize(cb);
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, cb, &devname[0], 0);
    //std::cout << "Device: " << devname.c_str() << "\n";   //打印使用的device名称

    
    
    /*
    Create a command queue(调用clCreateCommandQueue函数）一个设备device对应一个command queue。
    上下文conetxt将命令发送到设备对应的command queue，设备就可以按顺序执行命令队列里的命令。
    */

    cl_command_queue queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, 0);
    if (queue == 0)
    {
        std::cout << "Can't create command queue\n";
        clReleaseContext(context);
        return;
    }

    
    /*
    Create device buffers(调用clCreateBuffer函数）
　　　　Buffer中保存的是数据对象，就是设备执行程序需要的数据保存在其中。
　　　　Buffer由上下文conetxt创建，这样上下文管理的多个设备就会共享Buffer中的数据。
       所转的kernel有几个参数需要创建几个Buffer，另外再加上需要创建结果存储的Buffer，例：ClAddOne 只需要一个参数这里创建了cl_a,
       结果存放在创建的cl_res,此处注意其中数据类型也要相应修改，即sizeof(cl_int)，例如：若为float则为sizeof(cl_float)
    */
    const int DATA_SIZE = N;
    cl_mem cl_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * DATA_SIZE, const_cast<int *>(&in[0]), NULL);
    cl_mem cl_res = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * DATA_SIZE, NULL, NULL);
    if (cl_a == 0  || cl_res == 0)
    {
        std::cout << "Can't create OpenCL buffer\n";
        clReleaseMemObject(cl_a);
        clReleaseMemObject(cl_res);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return;
    }

    /*
    当有两个参数时，示例如下：
    cl_mem cl_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * DATA_SIZE, &a[0], NULL);
    cl_mem cl_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * DATA_SIZE, &b[0], NULL);
    cl_mem cl_res = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * DATA_SIZE, NULL, NULL);
    if(cl_a == 0 || cl_b == 0 || cl_res == 0) {
	    std::cout << "Can't create OpenCL buffer\n";
	    clReleaseMemObject(cl_a);
	    clReleaseMemObject(cl_b);
	    clReleaseMemObject(cl_res);
	    clReleaseCommandQueue(queue);
	    clReleaseContext(context);
	    return 0;
	}
    */



    //----------------------------以上都是执行OpenCL kernel的准备工作


    // const char *类型的kernel编写与编译
    const char *source = "__kernel void AddOneKernel(__global const int* a, __global int* b)  \
                            { \
                                int idx = get_global_id(0); \
                                b[idx] = a[idx] + 1; \
                            }";
    cl_program program = clCreateProgramWithSource(context, 1, &source, 0, 0);
    
    if (clBuildProgram(program, 0, 0, 0, 0, 0) != CL_SUCCESS)
    {
        std::cout << "Can't load or build program\n";
        clReleaseMemObject(cl_a);
        clReleaseMemObject(cl_res);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return;
    }

    //一個 OpenCL kernel 程式裡面可以有很多個函式。因此，還要取得程式中函式的進入點
    cl_kernel clkernel = clCreateKernel(program, "AddOneKernel", 0);   //引号中名称换为改写后的kernel名称
    if (clkernel == 0)
    {
        std::cout << "Can't load kernel\n";
        clReleaseProgram(program);
        clReleaseMemObject(cl_a);
        clReleaseMemObject(cl_res);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return;
    }

    
    //要执行行 kernel，只需要先設定好函式的參數，此函数只有两个参数
    clSetKernelArg(clkernel, 0, sizeof(cl_mem), &cl_a);
    clSetKernelArg(clkernel, 1, sizeof(cl_mem), &cl_res);

    
    size_t work_size = DATA_SIZE;

    //统计kernel运行时间相关信息
    int loop = 1000; // 迭代次数
    clock_t time_start = clock();
    cl_event *tt = new cl_event[loop];
    cl_int err_code, kerneltimer;
    for(int j = 0; j < loop; j++) {
        err = clEnqueueNDRangeKernel(queue, clkernel, 1, 0, &work_size, 0, 0, 0, &tt[j]);
        clFinish(queue);
    }
    clock_t time_end = clock();

    cl_ulong starttime = 0, endtime = 0;
    unsigned long elapsed = 0;
    for(int j = 0; j < loop; j++) {
        err_code = clGetEventProfilingInfo(tt[j], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &starttime, NULL);
        kerneltimer = clGetEventProfilingInfo(tt[j], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endtime, NULL);
        elapsed += (unsigned long)(endtime - starttime);
    }

    std::cout << "opencl time use:" << 1000 * (time_end - time_start) / (double)CLOCKS_PER_SEC << "us" << std::endl;
    std::cout << "opencl kernel time use:" << elapsed / 1000.0 / 1000 << "us" << std::endl;
    std::cout << std::endl;
    //由於執行的結果是在OpenCL裝置的内存中，所以要取得結果，需要把它的內容复制到CPU能存取的主存中，此处注意数据类型，如上145行说述
    if (err == CL_SUCCESS)
    {
        err = clEnqueueReadBuffer(queue, cl_res, CL_TRUE, 0, sizeof(int) * DATA_SIZE, &res[0], 0, 0, 0);
    }


    clReleaseKernel(clkernel);
    clReleaseProgram(program);
    clReleaseMemObject(cl_a);
    //clReleaseMemObject(cl_b);
    clReleaseMemObject(cl_res);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return;
}



//按照tensorflow提供的接口，把上述实现的整个OpenCL Kernel过程,能够在tensorflow框架下执行
class ClAddOneOp : public OpKernel {
 public:
  explicit ClAddOneOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {

    // 获取输入tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();  //相应数据类型进行修改，这个在对应的CUDA下kernel源码中也有对应借鉴即可，本例在cuda_op_kernel.cc中

    // 创建输出tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    auto output = output_tensor->template flat<int32>();    //对相应数据类型进行修改，这个在对应的CUDA下kernel源码中也有对应借鉴即可，同上



    const int N = input.size();

    //在tensorflow下具体执行的代码，即上述包括改写的OpenCl kernel的所有Opencl 程序
    ClKernelLauncher(input.data(), N, output.data());

  }
};


//把上述过程，即自定义的OP（kernel封装后的操作）注册到tensorflow下，以便在tensorflow下调用
//其中的Name与前边REGISTER_OP("ClAddOne")  保持一致，中间Device不用修改，最后一个flag命名格式：前边Name名字+Op
REGISTER_KERNEL_BUILDER(Name("NkAddOne").Device(DEVICE_CPU), ClAddOneOp);