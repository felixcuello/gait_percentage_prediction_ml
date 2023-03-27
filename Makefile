all: build up

build: build up

shell:
	docker compose exec gait_app bash

testshell:
	docker compose run gait_app bash

up:
	docker compose up

down:
	docker compose down

# -- This command initiates the docker agent in MacOS
macos_docker_up:
	open -a Docker
