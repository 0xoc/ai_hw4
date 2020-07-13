# AI HW #4
## Ali Parvizi 9632433

This assignment is done in python 3.7.
Install the requirements first

```
pip install -r requirements.py
```

## Loader
The ```loader.py``` file, includes a loader
 function that reads a dataset
  file and converts its samples into
   Sample objects.
   
   the loader function returns 
   an array of normalized objects.

## Part A
Part A of the assignment is implemented in ```partA.py``` file.
in this file, the two dataset files are loaded through the loader function from ```loader.py```.

Two important functions in this file are: ```gradient_descent``` and ```closed_form```.
which correspond to gradient descent and closed form linear regression methods respectively.

### How to Run
 - run ```python partA.py```

### What it does

 - by running the above command, partA.py will start executing.
4 plots will be generated one after another.
The first plot shows all the normalized data from dataset 1 and a line that has been learned through gradient decent.
After you close this plot, the second plot will show up. which is a plot of all the normalized data from dataset 1 and a 
line that has been learned through closed form equations.

 - This is how normalization is done: value / ( max_value - min_value )
 with the above formulation every sample data is normalized. If we don't normalize the data,
 different features with different scales and units may cause abnormalities in the final result.
 for example if one feature is measured in meters and another feature is measured in kilometers 
 and the value of the first one is 0.01 and the the value of the second one is 800,
 a 25 percent increase in each of them leads to the first value being 0.0125 and the second
 value being 1000, this means that the change in the second value dominates the behaviour of the whole system.
 

 - The 3rd and 4th plots are the same as the first two plots but with the data from dataset 2.

 - dataset 2 has some outliers. These outliers cause the learned model
to be a bit off and the accuracy of the model decreases because it's 
trying to account for all of the data including the outliers. In other words the
Least Square Objective function tries to minimize the error with respect to all of the samples.
now if some of the samples are far from the other samples, the learned method would be less accurate
for the majority of the samples.

 - the 5th plot shows theta 1 and theta 2 as they were learned by gradient decent method
 
 - the last plot shows a binary sigmoid function. A binary sigmoid function is used in classification problems.
 if the output of this function is more than 0.5, the input is considered to be of a class represented by 1 and
 if it's less than 0.5, it's considered to be of class 0
 
## Part B
 In locally weighted linear regression, instead of all data having the same importance, 
 the closer a sample is to a new unknown sample the more important it is.
 we could use a gaussian like weight function.

 - w<sup>i</sup> = exp(- (X<sup>i</sup> - X<sup>new</sup>)<sup>2</sup> / 2 )

according to the above weight function, the closer X<sup>i</sup> is to X<sup>new</sup>
the more weight it has thus it has more effect.
Now we will modify the normal linear regression cost function to include the weights.

 - J(&theta;) = &Sigma; w<sup>(i)</sup> (y<sup>(i)</sup> - (&theta;<sub>0</sub> + &theta;<sub>1</sub>x<sup>(i)</sup>) )<sup>2</sup>

Now we need to find argmin(&theta;).

 - &part;J / &part;&theta;<sub>0</sub> = 
-2 &Sigma; w<sup>(i)</sup> (y<sup>(i)</sup> - (&theta;<sub>0</sub> + &theta;<sub>1</sub>x<sup>(i)</sup>) )
 - &part;J / &part;&theta;<sub>1</sub> = 
-2 &Sigma; w<sup>(i)</sup> (y<sup>(i)</sup> - (&theta;<sub>0</sub> + &theta;<sub>1</sub>x<sup>(i)</sup>) ) x<sup>(i)</sup>

Equating the above formulas to zero, we will have:

 - &Sigma; w<sup>(i)</sup>&theta;<sub>0</sub> + &Sigma;w<sup>(i)</sup>
&theta;<sub>1</sub>x<sup>(i)</sup> = &Sigma;w<sup>(i)</sup>y<sup>(i)</sup>

 - &Sigma; w<sup>(i)</sup>&theta;<sub>0</sub> + &Sigma;w<sup>(i)</sup>
&theta;<sub>1</sub>x<sup>(i)</sup>x<sup>(i)</sup> = &Sigma;w<sup>(i)</sup>y<sup>(i)</sup>x<sup>(i)</sup>

it can be rewritten as:

![hi](http://serve.l37.ir/eq.png)

in other words:
A&theta; = b

**now we can solve for &theta; and arrive at a closed form formula:
&theta; = A<sup>-1</sup>b**