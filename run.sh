# This file contains the instructions to compile and run a C file

# Compile the file
## Get current file name
file_name=$(basename "$1")

## Find all C files and remove the main file and all test files
c_files=$(ls *.c | grep -v 'main' | grep -v 'test')

## Check if build directory exists, if not create it
if [ ! -d "build" ]; then
  mkdir build
fi

## Compile all C files
gcc -lm -o "./build/test" $c_files $file_name

## Run the compiled file
./build/test