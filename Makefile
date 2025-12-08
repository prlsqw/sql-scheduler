.PHONY: language scheduler language-lib scheduler-lib clean format

language:
	clang -o language-test language/tests.c language/executor.c language/parser.c language/utils.c

language-lib:
	clang -c language/executor.c -o language/executor.o -I language/headers
	clang -c language/parser.c -o language/parser.o -I language/headers
	clang -c language/utils.c -o language/utils.o -I language/headers

scheduler:
	clang -o scheduler-test scheduler/tests.c scheduler/scheduler.c scheduler/job_queue.c language/executor.c language/utils.c -I scheduler/headers

scheduler-lib:
	make language-lib
	clang -c scheduler/scheduler.c -o scheduler/scheduler.o -I scheduler/headers
	clang -c scheduler/job_queue.c -o scheduler/job_queue.o -I scheduler/headers

clean:
	rm -f language-test
	rm -f scheduler-test
	rm -f **/*.o

format:
	find . -type f \( -name "*.c" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \) -exec clang-format -i {} +