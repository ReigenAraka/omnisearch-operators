import grpc
from concurrent import futures
import rpc_pb2, rpc_pb2_grpc
from xception import Xception, run

ENDPOINT = "127.0.0.1:50012"


class OperatorServicer(rpc_pb2_grpc.OperatorServicer):
    def __init__(self):
        pass

    def Execute(self, request, context):
        grpc_vectors = []
        vectors = run(request.datas, request.urls)
        for vector in vectors:
            v = rpc_pb2.Vector(element=vector)
            grpc_vectors.append(v)
        return rpc_pb2.ExecuteReply(nums=len(vectors),
                                    vectors=grpc_vectors,
                                    metadata=[])

    def Healthy(self, request, context):
        return rpc_pb2.HealthyReply(healthy="healthy")

    def Identity(self, request, context):
        xception = Xception()
        return rpc_pb2.IdentityReply(name=xception.name,
                                     endpoint=ENDPOINT,
                                     type=xception.type,
                                     input=xception.input,
                                     output=xception.output,
                                     dimension=xception.dimension,
                                     metricType=xception.metric_type)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rpc_pb2_grpc.add_OperatorServicer_to_server(OperatorServicer(), server)
    server.add_insecure_port('[::]:50012')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
