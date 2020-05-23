COMMIT_ID = $(shell git rev-parse --short HEAD)
GIT_TAG = $(shell git tag --points-at HEAD)
IMAGE_NAME = custom-operator
TEST_PORT = 53001

ifeq ($(GIT_TAG), )
  TAG=$(COMMIT_ID)
else
  TAG=$(GIT_TAG)
endif

cpu:
	docker build  \
 	--build-arg base_tag=ubuntu:bionic-20200219 \
 	-t zilliz/$(IMAGE_NAME):$(TAG) \
 	-f Dockerfile .

gpu:
	docker build  \
 	--build-arg base_tag=nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 \
 	-t zilliz/$(IMAGE_NAME)-gpu:$(TAG) \
 	-f Dockerfile-gpu .

test-cpu:
	docker run -p $(TEST_PORT):$(TEST_PORT) \
	-e OP_ENDPOINT=127.0.0.1:$(TEST_PORT) -v `pwd`/tmp:/app/tmp \
	--name "$(IMAGE_NAME)-cpu-test" -d zilliz/$(IMAGE_NAME):$(TAG)

	echo "sleep 15s for waiting container to init and warmup" && sleep 15s
	python3 ../test_grpc.sh.py -e 127.0.0.1:$(TEST_PORT) || echo "[ERROR] test grpc failed"
	docker rm -f "$(IMAGE_NAME)-cpu-test"

test-gpu:
	docker run -p $(TEST_PORT):$(TEST_PORT) \
	--gpus="device=0" -e device_id="/device:GPU:0" \
	-e OP_ENDPOINT=127.0.0.1:$(TEST_PORT) -v `pwd`/tmp:/app/tmp \
	--name "$(IMAGE_NAME)-gpu-test" -d zilliz/$(IMAGE_NAME)-gpu:$(TAG)

	echo "sleep 15s for waiting container to init and warmup" && sleep 15s
	python3 ../test_grpc.sh.py -e 127.0.0.1:$(TEST_PORT) || echo "[ERROR] test grpc failed"
	docker rm -f "$(IMAGE_NAME)-gpu-test"
