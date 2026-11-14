# hw2-leveldb

这是一个高级计算机系统课程的作业项目，专注于LevelDB数据库的性能测试和优化。项目通过一系列基准测试任务，探索LevelDB在不同配置下的性能表现，包括块大小、布隆过滤器、写缓冲区大小、延迟和缓存机制等。

## 项目结构

- `src/`: 源代码目录，包含C++实现的基准测试程序
- `scripts/`: 运行脚本，用于执行各项任务
- `data/`: 原始测试数据
- `results/`: 处理后的结果数据，包括图表和表格
- `report/`: 项目报告
- `third_party/leveldb/`: LevelDB源码
- `db/`: 数据库文件存储目录

## 任务概述

### Task 1: 配置性能测试
测试不同LevelDB配置对性能的影响：
- 块大小 (4KB, 16KB, 64KB)
- 布隆过滤器开关
- 写缓冲区大小 (4MB, 16MB, 64MB)

### Task 2: 延迟基准测试
测量LevelDB的读写延迟特性。

### Task 3: 缓存基准测试

评估缓存机制对性能的影响，使用Zipf分布模拟访问模式。

## 构建和运行

### 依赖项
- C++编译器 (g++)
- LevelDB库
- Python (用于数据处理和绘图)

### 构建
```bash
make
```

### 运行任务
使用scripts目录下的脚本：
```bash
./scripts/run_task1.sh  # 运行Task 1
./scripts/run_task2.sh  # 运行Task 2
./scripts/run_task3.sh  # 运行Task 3
```

### 菜单
运行主菜单：
```bash
./scripts/menu.sh
```

## 结果分析

运行脚本后，结果将保存在`results/`目录中，包括：
- CSV表格数据
- 生成的图表
- 性能汇总报告

使用Python脚本处理和可视化结果：
```bash
python3 scripts/plot_task1_results.py
python3 scripts/plot_task2_results.py
python3 scripts/plot_task3_results.py
```

## 报告

详细的项目报告请查看`report/report.md`。