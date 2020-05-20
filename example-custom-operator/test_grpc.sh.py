import grpc
import rpc.rpc_pb2 as pb
import rpc.rpc_pb2_grpc as rpc_pb2_grpc


endpoint = '127.0.0.1:52001'


def identity(endpoint):
    try:
        with grpc.insecure_channel(endpoint) as channel:
            stub = rpc_pb2_grpc.OperatorStub(channel)
            res = stub.Identity(pb.IdentityRequest())
            return {
                "name": res.name,
                "endpoint": res.endpoint,
                "type": res.type,
                "input": res.input,
                "output": res.output,
                "dimension": res.dimension,
                "metric_type": res.metricType
            }
    except Exception as e:
        raise e


def health(endpoint):
    try:
        with grpc.insecure_channel(endpoint) as channel:
            stub = rpc_pb2_grpc.OperatorStub(channel)
            res = stub.Healthy(pb.HealthyRequest())
            return res.healthy
    except Exception as e:
        raise e


def execute(endpoint, datas=[], urls=[]):
    try:
        with grpc.insecure_channel(endpoint) as channel:
            stub = rpc_pb2_grpc.OperatorStub(channel)
            res = stub.Execute(pb.ExecuteRequest(urls=urls, datas=datas))
            return [list(x.element) for x in res.vectors], res.metadata
    except Exception as e:
        raise e


if __name__ == '__main__':
    test_url = 'http://a3.att.hudong.com/14/75/01300000164186121366756803686.jpg'
    print(identity(endpoint))
    print(health(endpoint))
    print(execute(endpoint, urls=[test_url]))
