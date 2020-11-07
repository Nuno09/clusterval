clean-build:
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

upload-test:
	python setup.py sdist
	twine upload --repository testpypi dist/*

upload:
	python setup.py sdist
	python3 -m twine upload dist/*

init:
	pip3 install -r requirements.txt


.PHONY: init test
