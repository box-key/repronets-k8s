.PHONY: test-trf test-phs test-all build-phs build-trf deploy-phs deploy-trf check

test-trf:
	@for i in ara chi heb jpn kor rus; do \
		curl --request GET -H "Host: repronet-trf-$$i.repronets.10.103.195.76.sslip.io" --url 'http://192.168.49.2:31388/predict?input=rocky&beam=3';\
		echo "";\
	done

test-phs:
	@for i in ara chi heb jpn kor rus; do \
		curl --request GET -H "Host: repronet-phs-$$i.repronets.10.103.195.76.sslip.io" --url 'http://192.168.49.2:31388/predict?input=rocky&beam=3';\
		echo "";\
	done

test-backend:
	curl --request GET -H "Host: repronet-backend.repronets.10.103.195.76.sslip.io" --url 'http://192.168.49.2:31388/predict?language=jpn&input=rocky&beam=3&model=phs'
	@echo ""
	curl --request GET -H "Host: repronet-backend.repronets.10.103.195.76.sslip.io" --url 'http://192.168.49.2:31388/predict?input=rocky&language=kor&beam=3&model=transformer'
	@echo ""

test-all: test-trf test-phs test-backend

build-phs:
	@for i in ara chi heb jpn kor rus; do \
		echo "Building $i";\
		docker build\
		-t dev.local/repronet-phs-$i\
		-f models/phonetisaurus/Dockerfile\
		--build-arg LANGUAGE=$i\
		. ;
	done

build-trf:
	@for i in ara chi heb jpn kor rus; do \
		echo "Building $$i";\
		docker build\
		-t dev.local/repronet-trf-$$i\
		-f models/transformer/Dockerfile\
		--build-arg LANGUAGE=$$i\
		. ;\
	done

deploy-phs:
	@for i in ara chi heb jpn kor rus; do\
		echo "Deploying $$i";\
		kubectl -n repronets apply -f models/phonetisaurus/components/ksvc_$$i.yaml;\
	done

deploy-trf:
	@for i in ara chi heb jpn kor rus; do \
		echo "Deploying $$i";\
		kubectl -n repronets apply -f models/transformer/components/ksvc_$$i.yaml;\
	done

check:
	kubectl -n repronets get ksvc
	kubectl -n repronets get po

run-test-pod:
	kubectl run --rm -it test --image=praqma/network-multitool bash
