
all: package

swig:
	swig -c++ -python swig/tpSDOs.i
	mv swig/tpSDOs.py swig/__init__.py

package: swig
	tar cJf tpSDOs.tar.xz contrib cpp python swig README.md LICENSE setup.py

clean:
	rm -rf build/

.PHONY: swig
