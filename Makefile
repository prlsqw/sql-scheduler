.PHONY: language scheduler language-lib scheduler-lib logger-lib clean format gen

all:
	make scheduler-lib
	clang -o orchestrator orchestrator.c \
		language/executor.o language/parser.o language/utils.o \
		scheduler/scheduler.o scheduler/job_queue.o scheduler/secretary.o scheduler/weights.o

language:
	clang -o language-test language/tests.c language/executor.c language/parser.c language/utils.c

language-lib:
	clang -c language/executor.c -o language/executor.o
	clang -c language/parser.c -o language/parser.o
	clang -c language/utils.c -o language/utils.o

scheduler:
	clang -o scheduler-test scheduler/tests.c scheduler/scheduler.c scheduler/secretary.c scheduler/weights.c scheduler/job_queue.c language/executor.c language/parser.c language/utils.c

scheduler-lib:
	make language-lib
	clang -c scheduler/scheduler.c -o scheduler/scheduler.o
	clang -c scheduler/job_queue.c -o scheduler/job_queue.o
	clang -c scheduler/secretary.c -o scheduler/secretary.o
	clang -c scheduler/weights.c -o scheduler/weights.o

logger-lib:
	clang -c logger/logger.c -o logger/logger.o

logger-test:
	clang -o logger-test logger/tests.c logger/logger.c

gen:
	nvcc -o dataset-gen gen/dataset-gen.cu language/utils.c
	nvcc -o queries-gen gen/queries-gen.cu language/executor.c language/utils.c

clean:
	rm -f language-test
	rm -f scheduler-test
	rm -f logger-test
	rm -f test_log.csv
	rm -f **/*.o

format:
	find . -type f \( -name "*.c" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \) -exec clang-format -i {} +