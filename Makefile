.PHONY: language scheduler language-lib scheduler-lib clean format gen

# TODO: add scheduler's secretary and weights file once PR is merged
all:
	make scheduler-lib
	clang -o orchestrator orchestrator.c \
		language/executor.o language/parser.o language/utils.o \
		scheduler/scheduler.o scheduler/job_queue.o

language:
	clang -o language-test language/tests.c language/executor.c language/parser.c language/utils.c

language-lib:
	clang -c language/executor.c -o language/executor.o
	clang -c language/parser.c -o language/parser.o
	clang -c language/utils.c -o language/utils.o

scheduler:
	clang -o scheduler-test scheduler/tests.c scheduler/scheduler.c scheduler/job_queue.c language/executor.c language/parser.c language/utils.c

scheduler-lib:
	make language-lib
	clang -c scheduler/scheduler.c -o scheduler/scheduler.o
	clang -c scheduler/job_queue.c -o scheduler/job_queue.o

gen:
	nvcc -o dataset-gen gen/dataset-gen.cu language/utils.c
	nvcc -o queries-gen gen/queries-gen.cu language/executor.c language/utils.c

clean:
	rm -f language-test
	rm -f scheduler-test
	rm -f **/*.o

format:
	find . -type f \( -name "*.c" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \) -exec clang-format -i {} +