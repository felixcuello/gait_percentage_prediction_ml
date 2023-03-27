all: build up

build:
	docker compose build --no-cache

shell:
	docker compose exec gait_app bash

testshell:
	docker compose run gait_app bash

up:
	docker compose up

down:
	docker compose down

mat2csv:
	docker-compose run --entrypoint bash gait_app /app/scripts/mat2csv.sh

# -- This command initiates the docker agent in MacOS
macos_docker_up:
	open -a Docker
