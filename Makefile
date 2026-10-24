.PHONY: all build run clean

CC = gcc
CFLAGS = -O3 -Wall
LDFLAGS = -lm
SRC = $(wildcard src/*.c)
BIN_DIR = bin
TARGET = $(BIN_DIR)/sinenet

all: build run

build:
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

run: build
	python demo.py

clean:
	rm -rf $(BIN_DIR) demo_results