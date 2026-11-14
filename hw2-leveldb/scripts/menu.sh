#!/bin/bash

set -e

echo "========================================"
echo " HW2 LevelDB Experiment Menu"
echo "========================================"
echo "1) run task1 formal + plot"
echo "2) run task2 formal + plot"
echo "3) run task3 formal + plot"
echo "4) run all formal tasks + plot"
echo "5) run task2 debug + plot"
echo "6) run task3 debug + plot"
echo "7) plot all existing results only"
echo "0) exit"
echo "========================================"

read -p "Choose an option: " choice

case "$choice" in
    1)
        make run-task1
        ;;
    2)
        make run-task2
        ;;
    3)
        make run-task3
        ;;
    4)
        make run-all
        ;;
    5)
        make run-task2-debug
        ;;
    6)
        make run-task3-debug
        ;;
    7)
        make plot
        ;;
    0)
        echo "Exit."
        ;;
    *)
        echo "Invalid option."
        exit 1
        ;;
esac
