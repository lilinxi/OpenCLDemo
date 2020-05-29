#include <iostream>
#include <chrono>
#include <OpenCL/opencl.h>

using namespace std;

void check_result(const int *buf, const int len) {
    int i;
    for (i = 0; i < len; i++) {
        if (buf[i] != (i + 1) * 2) {
            cout << "Result error!" << endl;
            break;
        }
    }
    if (i == len) cout << "Result ok." << endl;
}

void init_buf(int *buf, int len) {
    int i;
    for (i = 0; i < len; i++) {
        buf[i] = i + 1;
    }
}

void equal_function(int *buf, int len) {
    for (int i = 0; i < len; i++) {
        buf[i] += buf[i];
    }
}


int main() {
    auto start = std::chrono::system_clock::now();
    cl_int ret;
    /** step 1: get platform */
    cl_uint num_platforms;
    ret = clGetPlatformIDs(0, NULL, &num_platforms); // get platform number
    if ((CL_SUCCESS != ret) || (num_platforms < 1)) {
        cout << "Error getting platform number: " << ret << endl;
        return 0;
    }
    cout << "platform number: " << num_platforms << endl;
    cl_platform_id platform_id = NULL;
    ret = clGetPlatformIDs(1, &platform_id, NULL); // get first platform id
    if (CL_SUCCESS != ret) {
        cout << "Error getting platform id: " << ret << endl;
        return 0;
    }
    /** step 2: get device */
    cl_uint num_devices;
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if ((CL_SUCCESS != ret) || (num_devices < 1)) {
        cout << "Error getting GPU device number: " << ret << endl;
        return 0;
    }
    cout << "GPU device number: " << num_devices << endl;
    cl_device_id device_id = NULL;
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (CL_SUCCESS != ret) {
        cout << "Error getting GPU device id: " << ret << endl;
        return 0;
    }
    /** step 3: create context */
    cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform_id, 0};
    cl_context context = NULL;
    context = clCreateContext(props, 1, &device_id, NULL, NULL, &ret);
    if ((CL_SUCCESS != ret) || (NULL == context)) {
        cout << "Error creating context: " << ret << endl;
        return 0;
    }
    /** step 4: create command queue */
    cl_command_queue command_queue = NULL;
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    if ((CL_SUCCESS != ret) || (NULL == command_queue)) {
        cout << "Error creating command queue: " << ret << endl;
        return 0;
    }
    /** step 5: create memory object */
    cl_mem mem_obj = NULL;
    int *host_buffer = NULL;
    const int ARRAY_SIZE = 100000000;
    const int BUF_SIZE = ARRAY_SIZE * sizeof(int);
    // create and init host buffer
    host_buffer = (int *) malloc(BUF_SIZE);
    init_buf(host_buffer, ARRAY_SIZE);
    // create opencl memory object using host ptr
    mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, BUF_SIZE, host_buffer, &ret);
    if ((CL_SUCCESS != ret) || (NULL == mem_obj)) {
        cout << "Error creating command queue: " << ret << endl;
        return 0;
    }
    /** step 6: create program */
    char *kernelSource =
            "__kernel void test(__global int *pInOut)\n"
            "{\n"
            " int index = get_global_id(0);\n"
            " pInOut[index] += pInOut[index];\n"
            "}\n";
    cl_program program = NULL;
    // create program
    program = clCreateProgramWithSource(context, 1, (const char **) &kernelSource, NULL, &ret);
    if ((CL_SUCCESS != ret) || (NULL == program)) {
        cout << "Error creating program: " << ret << endl;
        return 0;
    }
    // build program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (CL_SUCCESS != ret) {
        cout << "Error building program: " << ret << endl;
        return 0;
    }
    /** step 7: create kernel */
    cl_kernel kernel = NULL;
    kernel = clCreateKernel(program, "test", &ret);
    if ((CL_SUCCESS != ret) || (NULL == kernel)) {
        cout << "Error creating kernel: " << ret << endl;
        return 0;
    }
    /** step 8: set kernel arguments */
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &mem_obj);
    if (CL_SUCCESS != ret) {
        cout << "Error setting kernel argument: " << ret << endl;
        return 0;
    }
    /** step 9: set work group size */
    cl_uint work_dim = 3; // in most opencl device, max dimition is 3
    size_t global_work_size[] = {ARRAY_SIZE, 1, 1};
    size_t *local_work_size = NULL; // let opencl device determine how to break work items into work groups
    /** step 10: run kernel */
    ret = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL,
                                 NULL);
    if (CL_SUCCESS != ret) {
        cout << "Error enqueue NDRange: " << ret << endl;
        return 0;
    }
    /** step 11: get result */
    int *device_buffer = (int *) clEnqueueMapBuffer(command_queue, mem_obj, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
                                                    BUF_SIZE, 0, NULL, NULL, &ret);
    if ((CL_SUCCESS != ret) || (NULL == device_buffer)) {
        cout << "Error map buffer: " << ret << endl;
        return 0;
    }
    auto stop = std::chrono::system_clock::now();
    std::cout << "cost: " << std::chrono::duration_cast<std::chrono::seconds>(stop - start).count()
              << " seconds\n";
    // check result
    check_result(device_buffer, ARRAY_SIZE);
    start = std::chrono::system_clock::now();
    equal_function(device_buffer, ARRAY_SIZE);
    stop = std::chrono::system_clock::now();
    std::cout << "cost: " << std::chrono::duration_cast<std::chrono::seconds>(stop - start).count()
              << " seconds\n";
    /** step 12: release all resources */
    if (NULL != kernel) clReleaseKernel(kernel);
    if (NULL != program) clReleaseProgram(program);
    if (NULL != mem_obj) clReleaseMemObject(mem_obj);
    if (NULL != command_queue) clReleaseCommandQueue(command_queue);
    if (NULL != context) clReleaseContext(context);
    if (NULL != host_buffer) free(host_buffer);
    return 0;
}