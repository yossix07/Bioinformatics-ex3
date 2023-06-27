# Bioinformatics-ex3

## Getting Started
- Clone this repositoy.

- You are all set! you have 2 options in order to run - 

## Run Executable
From the cloned folder:

- #### in order to run the runnet0 program, you can run the following command - 
<br/> $ runnet0.exe <weights_file> <data_file>
<br/> example run -
<br/> $ runnet0.exe wnet0.txt someTestFile.txt
<br/> where someTestFile.txt is a file with unlabeled inputs.
<br/> the program will create a file name "output0.txt" with the predicated labels.

- #### in order to run the runnet1 program, you can run the following command - 
<br/> $ runnet1.exe <weights_file> <data_file>
<br/> example run -
<br/> $ runnet1.exe wnet1.txt someTestFile.txt
<br/> where someTestFile.txt is a file with unlabeled inputs.
<br/> the program will create a file name "output1.txt" with the predicated labels.

- #### in order to run the buildnet0 program, you can run the following command - 
<br/> $ buildnet0.exe <learning_file> <test_file>
<br/> example run -
<br/> $ buildnet0.exe nn0_train.txt nn0_test.txt
<br/> Note that the files nn0_train.txt and nn0_test.txt exist in the repo and we used them while running buildnet0.

- #### in order to run the buildnet1 program, you can run the following command - 
<br/> $ buildnet1.exe <learning_file> <test_file>
<br/> example run -
<br/> $ buildnet1.exe nn1_train.txt nn1_test.txt
<br/> Note that the files nn1_train.txt and nn1_test.txt exist in the repo and we used them while running buildnet1.
 

## Run From Source Code
- From the cloned folder, install numpy using the command: $ pip install numpy

- #### in order to run the runnet0 program, you can run the following command - 
<br/> $ runnet0.py <weights_file> <data_file>
<br/> example run -
<br/> $ runnet0.py wnet0.txt someTestFile.txt
<br/> where someTestFile.txt is a file with unlabeled inputs.
<br/> the program will create a file name "output0.txt" with the predicated labels.

- #### in order to run the runnet1 program, you can run the following command - 
<br/> $ runnet1.py <weights_file> <data_file>
<br/> example run -
<br/> $ runnet1.py wnet1.txt someTestFile.txt
<br/> where someTestFile.txt is a file with unlabeled inputs.
<br/> the program will create a file name "output1.txt" with the predicated labels.

- #### in order to run the buildnet0 program, you can run the following command - 
<br/> $ buildnet0.py <learning_file> <test_file>
<br/> example run -
<br/> $ buildnet0.py nn0_train.txt nn0_test.txt
<br/> Note that the files nn0_train.txt and nn0_test.txt exist in the repo and we used them while running buildnet0.

- #### in order to run the buildnet1 program, you can run the following command - 
<br/> $ buildnet1.py <learning_file> <test_file>
<br/> example run -
<br/> $ buildnet1.py nn1_train.txt nn1_test.txt
<br/> Note that the files nn1_train.txt and nn1_test.txt exist in the repo and we used them while running buildnet1.
