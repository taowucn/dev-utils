CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra
CXXFLAGS = -Wall -Wextra -std=c++11
TARGET = test.elf
OBJS = obj/main.o obj/sum.o
PARSER_CPP_BIN = out/elf_parser_cpp
PARSER_C_BIN = out/elf_parser_c

.PHONY: all run clean parse_py  parser_cpp parser_c

all: run parse_py

$(TARGET): $(OBJS)
	mkdir -p obj out
	$(CC) $(OBJS) -o $@

$(PARSER_CPP_BIN): elf_parser.cpp
	mkdir -p obj out
	$(CXX) $(CXXFLAGS) $< -o $@

$(PARSER_C_BIN): elf_parser.c
	mkdir -p obj out
	$(CC) $(CFLAGS) $< -o $@

obj/%.o: src/%.c
	mkdir -p obj
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

parse_py: $(TARGET)
	python3 elf_parser.py -f $(TARGET) -m 0

parse_cpp: $(TARGET) $(PARSER_CPP_BIN)
	./$(PARSER_CPP_BIN) -f $(TARGET)

parse_c: $(TARGET) $(PARSER_C_BIN)
	./$(PARSER_C_BIN) -f $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET) $(PARSER_CPP_BIN) $(PARSER_C_BIN)
	rm -rf obj out
