CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra
CXXFLAGS = -Wall -Wextra -std=c++11
TARGET = test.elf
OBJS = obj/main.o obj/sum.o

PARSER_CPP_BIN = out/elf_parser_cpp
PARSER_C_BIN = out/elf_parser_c
PARSER_LIBELF_BIN = out/elf_parser_bylib_c
PARSER_CPP_LIBELF_BIN = out/elf_parser_bylib_cpp

.PHONY: all run clean parse_py parser_cpp parser_c parser_libelf parser_cpp_libelf

all: run parse_py

pre:
	mkdir -p obj out

$(TARGET): $(OBJS) pre
	$(CC) $(OBJS) -o $@

$(PARSER_CPP_BIN): elf_parser.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

$(PARSER_C_BIN): elf_parser.c pre
	$(CC) $(CFLAGS) $< -o $@

$(PARSER_LIBELF_BIN): elf_parser_bylib.c pre
	$(CC) $(CFLAGS) $< -o $@ -lelf

$(PARSER_CPP_LIBELF_BIN): elf_parser_bylib.cpp pre
	$(CXX) $(CXXFLAGS) $< -o $@ -lelf

obj/%.o: src/%.c pre
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

parse_py: $(TARGET)
	python3 elf_parser.py -f $(TARGET) -m 0

parse_cpp: $(TARGET) $(PARSER_CPP_BIN)
	./$(PARSER_CPP_BIN) $(TARGET)

parse_c: $(TARGET) $(PARSER_C_BIN)
	./$(PARSER_C_BIN) $(TARGET)

parse_libelf_c: $(TARGET) $(PARSER_LIBELF_BIN)
	./$(PARSER_LIBELF_BIN) $(TARGET)

parse_libelf_cpp: $(TARGET) $(PARSER_CPP_LIBELF_BIN)
	./$(PARSER_CPP_LIBELF_BIN) $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET) $(PARSER_CPP_BIN) $(PARSER_C_BIN) $(PARSER_LIBELF_BIN) $(PARSER_CPP_LIBELF_BIN)
	rm -rf obj out
