
# How to test grpc server
一般测试 grpc 服务有以下两种方法:
1. 使用 grpcurl 工具
2. 编写 grpc 客户端
## 使用 grpcurl 工具 
[如何安装和使用 grpcurl](https://github.com/fullstorydev/grpcurl)

## 编写 grpc 客户端
针对 omnisearch 的 grpc 接口, 在本仓库中提供了相关的客户端程序, 以便减小开发者的测试成本.
具体代码可参考```test_grpc.sh```

### 使用前提
请确保运行环境满足以下要求:
- python3
- pip3
- grpc

### 使用方式
```bash
python3 test_grpc.sh -e ${server_ip}:${server_port}
```

运行结果:
附结果图片