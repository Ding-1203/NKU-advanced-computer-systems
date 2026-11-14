CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

ROOT := $(HOME)/advanced-computer-systems/hw2-leveldb

LEVELDB_DIR := $(ROOT)/third_party/leveldb
LEVELDB_BUILD := $(LEVELDB_DIR)/build
LEVELDB_INCLUDE := $(LEVELDB_DIR)/include
LEVELDB_LIB := $(LEVELDB_BUILD)

INCLUDES := -I$(ROOT)/src -I$(LEVELDB_INCLUDE)
LIBS := -L$(LEVELDB_LIB) -lleveldb -lpthread

BIN_DIR := $(ROOT)/bin

TASK2_BIN := $(BIN_DIR)/task2_latency_bench
TASK3_BIN := $(BIN_DIR)/task3_cache_bench
TEST_BIN := $(BIN_DIR)/test_cache_zipf

TASK2_SRC := $(ROOT)/src/task2_latency_bench.cpp
TASK3_SRC := $(ROOT)/src/task3_cache_bench.cpp
CACHE_SRC := $(ROOT)/src/kv_cache.cpp
ZIPF_SRC := $(ROOT)/src/zipf.cpp
TEST_SRC := $(ROOT)/src/test_cache_zipf.cpp

.PHONY: all setup build menu clean test \
	run-task1 run-task2 run-task3 run-all \
	run-task2-debug run-task3-debug \
	plot plot-task1 plot-task2 plot-task3

all: setup build menu

setup:
	@echo "Installing dependencies..."
	sudo apt update
	sudo apt install -y build-essential cmake git g++ make python3 python3-pip
	pip3 install pandas matplotlib
	@echo "Preparing LevelDB..."
	@if [ ! -d "$(LEVELDB_DIR)" ]; then \
		mkdir -p $(ROOT)/third_party; \
		cd $(ROOT)/third_party && git clone https://github.com/google/leveldb.git; \
	fi
	cd $(LEVELDB_DIR) && git submodule update --init --recursive
	mkdir -p $(LEVELDB_BUILD)
	cd $(LEVELDB_BUILD) && cmake .. && make -j$$(nproc)
	@echo "Setup finished."

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

build: $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(TASK2_SRC) $(INCLUDES) $(LIBS) -o $(TASK2_BIN)
	$(CXX) $(CXXFLAGS) $(TASK3_SRC) $(CACHE_SRC) $(ZIPF_SRC) $(INCLUDES) $(LIBS) -o $(TASK3_BIN)
	@echo "Build finished."

menu:
	./scripts/menu.sh

run-task1:
	./scripts/run_task1.sh
	python3 scripts/parse_task1_results.py
	python3 scripts/plot_task1_results.py

run-task2: build
	./scripts/run_task2.sh formal
	python3 scripts/plot_task2_results.py

run-task3: build
	./scripts/run_task3.sh formal
	python3 scripts/plot_task3_results.py

run-task2-debug: build
	./scripts/run_task2.sh debug
	python3 scripts/plot_task2_results.py

run-task3-debug: build
	./scripts/run_task3.sh debug
	python3 scripts/plot_task3_results.py

run-all: run-task1 run-task2 run-task3

plot: plot-task1 plot-task2 plot-task3

plot-task1:
	python3 scripts/parse_task1_results.py
	python3 scripts/plot_task1_results.py

plot-task2:
	python3 scripts/plot_task2_results.py

plot-task3:
	python3 scripts/plot_task3_results.py

test: $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(TEST_SRC) $(CACHE_SRC) $(ZIPF_SRC) $(INCLUDES) -o $(TEST_BIN)
	$(TEST_BIN)

clean:
	rm -rf $(BIN_DIR)
