To build the code, run the following commands from your terminal:

```shell
$ cd resnet
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/home/testroot/CNN/libtorch-cxx11 ..
$ make
```

Execute the compiled binary to train the model:

```shell
$ ./resnet
```

