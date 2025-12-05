.PHONY: language

language:
	clang -o lang language/tests.c language/executor.c language/parser.c language/utils.c

clean:
	rm -f lang
	rm -f *.o

format:
	clang-format -i *.c *.h