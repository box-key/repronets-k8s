.PHONY: listen ping


listen:
	# NAME is pod's name
	$(eval NAME=sklearn-iris-predictor-default-00001-deployment-5c9988d847gjp49)
	kubectl port-forward -n kserve-test $(NAME) 8080:8080

ping:
	# run this target while listen is run
	curl -v http://localhost:8080/v1/models/sklearn-iris:predict -d @./test.json

get-kserve-pods:
	kubectl get inferenceservices -n kserve-test


get-all:
	kubectl get $(x) --all-namespaces

