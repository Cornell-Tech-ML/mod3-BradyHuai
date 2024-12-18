# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


## Task 3.1
### Map diagnostics output
![Alt text](./images/map-task1.png)
### Zip diagnostics output
![Alt text](./images/zip-task1-1.png)
![Alt text](./images/zip-task1-2.png)
### Reduce diagnostics output
![Alt text](./images/reduce-task1.png)


## Task 3.2
![Alt text](./images/task2-1.png)
![Alt text](./images/task2-2.png)


## Task 3.4
![Alt text](./images/task3.4.png)
![Alt text](./images/graph.png)

## Task 3.5

### Simple

#### GPU
![Alt text](./images/simple-gpu-1.png)
![Alt text](./images/simple-gpu-2.png)
![Alt text](./images/simple-gpu-3.png)
Time per epoch: 1.868s
#### CPU
![Alt text](./images/simple-cpu.png)
Time per epoch: 0.251s


### Split

#### GPU
![Alt text](./images/split-gpu-1.png)
![Alt text](./images/split-gpu-2.png)
Time per epoch: 1.816s
#### CPU
![Alt text](./images/split-cpu.png)
Time per epoch: 0.263s

### XOR

#### GPU
![Alt text](./images/xor-gpu-1.png)
![Alt text](./images/xor-gpu-2.png)
Time per epoch: 1.867s
#### CPU
![Alt text](./images/xor-cpu.png)
Time per epoch: 0.251s

### Simple Large Model (hidden size = 200)
#### GPU
![Alt text](./images/large-gpu-1.png)
![Alt text](./images/large-gpu-2.png)
Time per epoch: 2.394s

#### CPU
![Alt text](./images/large-cpu-1.png)
![Alt text](./images/large-cpu-2.png)
Time per epoch: 0.478s