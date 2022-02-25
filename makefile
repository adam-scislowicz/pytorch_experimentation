.PHONY: docker-image docker-interactive clean

docker-image: Dockerfile
	PROJECT_SLUG="pytorch_experimentation" GID=$(shell id -g) DOCKER_BUILDKIT=1 docker build -f $< -t pytorch_experimentation:latest .

docker-interactive:
	docker run -v $(PWD):/home/$(USER)/pytorch_experimentation -it -w /home/$(USER)/pytorch_experimentation pytorch_experimentation:latest

clean:
	@rm -rf build* cmake-build* src/rust/target src/python/pytorch_experimentation.egg-info
