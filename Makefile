# Local development helpers.
# Not deployment configuration: the deployed image is built from the Dockerfile.

.PHONY: run-local

# Start Maui locally on the Gunicorn/gevent runtime, with the activated
# virtualenv, bound to the local address 127.0.0.1:5000.
run-local:
	LOG_LEVEL=INFO gunicorn main:app -k gevent \
		--workers 1 --worker-connections 10 \
		--timeout 300 --bind 127.0.0.1:5000
