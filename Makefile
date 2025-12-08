.PHONY: language gen

language:
	clang -o lang language/tests.c language/executor.c language/parser.c language/utils.c

gen:
	nvcc -o dataset-gen gen/dataset-gen.cu language/utils.c
	nvcc -o queries-gen gen/queries-gen.cu language/executor.c language/utils.c

clean:
	rm -f lang
	rm -f *.o

format:
	find . -type f \( -name "*.c" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \) -exec clang-format -i {} +