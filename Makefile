.PHONY: language

language:
	clang -o lang language/test.c language/executor.c language/parser.c language/utils.c

clean:
	rm -f lang
	rm -f *.o