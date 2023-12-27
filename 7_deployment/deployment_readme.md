## 命令式编程与符号式编程

## Torchscript 
1. IR: intermediate representation
2. 静态图,由torch jit compiler编译
3. `trace`与`script`的区别,在于控制流的记录
4. 混合 `Scripting`和`Tracing`
5. 提供save和load的api.格式保存code,参数,属性,debug信息
6. 在生产中低延迟的需求,需要使用其他语言:c++

> step1: 将pytorch model转化成torchscript
> step2: 序列化script module 为文件
> step3: C++中加载scritp module( 依赖LibTorch)
> step4: 在c++中执行script module
> step5: 