import grpc
import rpc.rpc_pb2 as pb
import rpc.rpc_pb2_grpc as rpc_pb2_grpc
import getopt
import sys

endpoint = '127.0.0.1:50001'


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
    opts, args = getopt.getopt(sys.argv[1:], '-h-e:', ['help', 'endpoint='])
    for opt_name, opt_value in opts:
        if opt_name in ('-h', '--help'):
            print("[*] Help info")
            exit()
        if opt_name in ('-e', '--filename'):
            endpoint = opt_value
            print("[*] Endpoint is ", endpoint)
            continue

    print("Begin to test: endpoint-%s" % endpoint)
    print("Endpoint information: ", identity(endpoint))
    print("Endpoint health: ", health(endpoint))
    vector, data = execute(endpoint, urls=[test_url])
    print("Result :\n  vector size: %d;  data size: %d" % (len(vector), len(data)))
    if len(vector) > 0:
        print("  vector dim: ", len(vector[0]))

    print("All tests over.")
    exit()
