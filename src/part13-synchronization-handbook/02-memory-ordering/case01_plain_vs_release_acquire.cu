#include "common.cuh"

#include <cuda/atomic>

/*
 * Case 01：普通 load/store 与 release/acquire
 *
 * 测试目的：
 *   对比普通内存访问和带发布/获取语义的 atomic 访问在编译结果上的区别。
 *
 * 运行时验证：
 *   1. plain store 写入 1，plain load 应读到 1；
 *   2. release store 写入 2，acquire load 应读到 2。
 *   kernel 按默认 stream 顺序执行，所以这里只验证 API 和数据结果，不用它
 *   证明并发 producer/consumer 协议。
 *
 * PTX/SASS 观察：
 *   plain：   st.global / ld.global
 *   ordered：st.release.gpu / ld.acquire.gpu
 *   SASS 可能表现为 STRONG load/store、MEMBAR 或目标架构专用序列。
 *
 * 语义结论：
 *   release 发布它之前的访问；读取到该发布值的 acquire 约束其后的访问。
 */

 /*
 汇编长下面这样
.visible .entry case01_plain_store(
	.param .u64 .ptr .align 1 case01_plain_store_param_0
)
{
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<3>;
	.loc	1 25 0

	ld.param.u64 	%rd1, [case01_plain_store_param_0];
	.loc	1 26 3
	cvta.to.global.u64 	%rd2, %rd1;
	mov.b32 	%r1, 1;
	st.global.u32 	[%rd2], %r1;
	.loc	1 27 1
	ret;

}
  逐行解释一下：
  .param .u64 .ptr .align 1 case01_plain_store_param_0 表示kernel有一个参数，类型为u64，对齐方式为1，名字为case01_plain_store_param_0。
  .ptr表示指针，.align 1表示对齐方式为1，表示这个参数是一个指针，并且指针对齐方式为1。u64是表示参数为64长度的地址。
  .reg .b32 	%r<2> 表示注册了2个32位寄存器。分别是%r0和%r1。
  .reg .b64 	%rd<3> 表示注册了3个64位寄存器。分别是%rd1、%rd2和%rd3。
  .loc	1 25 0 表示代码中的行数。
  ld.param.u64 	%rd1, [case01_plain_store_param_0]; 表示从case01_plain_store_param_0中加载一个u64值到%rd1寄存器。flag是个地址，这里rd1就是保存这个地址。
  cvta.to.global.u64 	%rd2, %rd1; 表示将%rd1寄存器中的值转换为全局地址，存储到%rd2寄存器。cuda有不同的地址空间，传入的是generic pointer，需要转换为global pointer。
  mov.b32 	%r1, 1; 表示将立即数1赋值给%r1寄存器。
  st.global.u32 	[%rd2], %r1; 表示将%r1寄存器中的值存储到%rd2寄存器所指向的全局内存地址。这里rd2是全局地址，所以存储的是全局内存。写入的数据是32位的，所以是u32。
  ret; 表示返回。
  .loc表示代码的行数。
 */
extern "C" __global__ void case01_plain_store(int* flag) {
  *flag = 1;
}

extern "C" __global__ void case01_plain_load(const int* flag, int* observed) {
  *observed = *flag;
}

/*
汇编长下面这样：
	// .globl	case01_release_store
.visible .entry case01_release_store(
	.param .u64 .ptr .align 1 case01_release_store_param_0
)
{
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<2>;
	.loc	1 66 0

	ld.param.u64 	%rd1, [case01_release_store_param_0];
	.loc	1 68 3
	.loc	2 80 14, function_name $L__info_string0, inlined_at 1 68 3
	.loc	3 93 3, function_name $L__info_string1, inlined_at 2 80 14
	.loc	4 319 3, function_name $L__info_string2, inlined_at 3 93 3
	.loc	5 1184 3, function_name $L__info_string3, inlined_at 4 319 3
	.loc	5 858 3, function_name $L__info_string4, inlined_at 5 1184 3
	.loc	5 1172 5, function_name $L__info_string5, inlined_at 5 858 3
	.loc	5 941 1, function_name $L__info_string6, inlined_at 5 1172 5
	mov.b32 	%r1, 2;
	// begin inline asm
	st.release.gpu.b32 [%rd1],%r1;
	// end inline asm
	.loc	1 69 1
	ret;
  }

  .param .u64 .ptr .align 1 case01_release_store_param_0：描述64位地址的入参
  .reg .b32 	%r<2>：描述2个32位寄存器，分别是%r0和%r1。
  .reg .b64 	%rd<2>：描述2个64位寄存器，分别是%rd1和%rd2。
  .loc	1 66 0：表示代码中的行数。
  ld.param.u64 	%rd1, [case01_release_store_param_0]：表示从case01_release_store_param_0中加载一个u64值到%rd1寄存器。flag是个地址，这里rd1就是保存这个地址。
  .loc	1 68 3：表示代码中的行数。
  .loc	2 80 14, function_name $L__info_string0, inlined_at 1 68 3：表示代码中的行数。
  .loc	3 93 3, function_name $L__info_string1, inlined_at 2 80 14：表示代码中的行数。
  .loc	4 319 3, function_name $L__info_string2, inlined_at 3 93 3：表示代码中的行数。
  .loc	5 1184 3, function_name $L__info_string3, inlined_at 4 319 3：表示代码中的行数。
  .loc	5 858 3, function_name $L__info_string4, inlined_at 5 1184 3：表示代码中的行数。
  .loc	5 1172 5, function_name $L__info_string5, inlined_at 5 858 3：表示代码中的行数。
  .loc	5 941 1, function_name $L__info_string6, inlined_at 5 1172 5：表示代码中的行数。
  上面这一堆实际上表示调用链。
  mov.b32 	%r1, 2：表示将立即数2赋值给%r1寄存器。
  st.release.gpu.b32 [%rd1],%r1：表示将%r1寄存器中的值存储到%rd1寄存器所指向的全局内存地址。这里rd1是全局地址，所以存储的是全局内存。写入的数据是32位的，所以是b32。
  .gpu表示memory scope为device scope，对应cuda::thread_scope_device。release表示memory order为release。release的作用范围是它之前的所有内存操作。
// begin inline asm和// end inline asm之间的部分是内联汇编代码。
*/
extern "C" __global__ void case01_release_store(int* flag) {
  cuda::atomic_ref<int, cuda::thread_scope_device> atomic_flag(*flag);
  atomic_flag.store(2, cuda::memory_order_release);
}

/*
这里atomic_flag.load翻译成PTX是ld.acquire.gpu.b32 %r1,[%rd1];因为带着gpu这个memory scope，所以不需要做地址转换。
但是赋值给*observed这是个store，需要做地址转换。
cuda中默认指针属于generic address space，有点类似于统一虚拟地址空间的感觉，真正计算时需要转换为真实的地址空间，确定到底是在global、
share、local。
但是这里的case中acquire、release都指定了thread_scope_device，所以不需要做地址转换。
*/
extern "C" __global__ void case01_acquire_load(int* flag, int* observed) {
  cuda::atomic_ref<int, cuda::thread_scope_device> atomic_flag(*flag);
  *observed = atomic_flag.load(cuda::memory_order_acquire);
}

int main() {
  int* flag = make_device_int();
  int* observed = make_device_int();

  case01_plain_store<<<1, 1>>>(flag);
  case01_plain_load<<<1, 1>>>(flag, observed);
  finish_kernel();
  int status = require_equal("plain load/store", read_device_int(observed), 1);

  case01_release_store<<<1, 1>>>(flag);
  case01_acquire_load<<<1, 1>>>(flag, observed);
  finish_kernel();
  status |=
      require_equal("release/acquire", read_device_int(observed), 2);

  CUDA_CHECK(cudaFree(observed));
  CUDA_CHECK(cudaFree(flag));
  return status;
}
