CFLAGSLIBS = `pkg-config --cflags --libs opencv`

main:
	nvcc main.cu tools.cu pca.cu net.cu -std=c++11 -o out $< $(CFLAGSLIBS) 
exec:
	./out
