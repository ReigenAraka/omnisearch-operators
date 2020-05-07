MAKEFILE_PATH=$(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
protoc:
	echo $(CURRENT_DIR)
	python -m grpc_tools.protoc -I $(CURRENT_DIR)/vggrpc/ $(CURRENT_DIR)/vggrpc/rpc.proto --python_out=$(CURRENT_DIR)/vggrpc --grpc_python_out=$(CURRENT_DIR)/vggrpc
	protoc -I $(CURRENT_DIR)/vggrpc/ $(CURRENT_DIR)/vggrpc/rpc.proto --go_out=plugins=grpc:$(CURRENT_DIR)/vggrpc
# build: protoc
# 	env GOOS=linux GOARCH=amd64 go build -o main .
# 	docker build --no-cache -t milvus.io/asearch-plugin/vgg:0.1.0 .
