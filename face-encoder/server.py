import os
import grpc
from concurrent import futures
import rpc.rpc_pb2, rpc.rpc_pb2_grpc
from face_embedding import run, face_encoder as encoder

ENDPOINT = os.getenv("OP_ENDPOINT", "127.0.0.1:50004")


class OperatorServicer(rpc.rpc_pb2_grpc.OperatorServicer):
    def __init__(self):
        pass

    def Execute(self, request, context):
        grpc_vectors = []
        vectors = run(request.datas, request.urls)
        for vector in vectors:
            v = rpc.rpc_pb2.Vector(element=vector)
            grpc_vectors.append(v)
        return rpc.rpc_pb2.ExecuteReply(nums=len(vectors),
                                        vectors=grpc_vectors,
                                        metadata=[])

    def Healthy(self, request, context):
        return rpc.rpc_pb2.HealthyReply(healthy="healthy")

    def Identity(self, request, context):
        return rpc.rpc_pb2.IdentityReply(name=encoder.name,
                                         endpoint=ENDPOINT,
                                         type=encoder.type,
                                         input=encoder.input,
                                         output=encoder.output,
                                         dimension=encoder.dimension,
                                         metricType=encoder.metric_type)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rpc.rpc_pb2_grpc.add_OperatorServicer_to_server(OperatorServicer(), server)
    server.add_insecure_port('[::]:50004')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
