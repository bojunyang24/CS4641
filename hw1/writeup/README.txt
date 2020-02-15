Bojun Yang

My code is tested in the main method.
I import the data in the beginning then I have commented out code for 
duplicating features. I also append the vector of 1s before I call any function.
I also append a vector of 1s in apply_RFF_transform. My reasoning is stated in my follow up discussings for this piazza question.
https://piazza.com/class/k4x9jo9fhkz18r?cid=17

After appending 1s, I calculate the theoretical data with lstsq.
I have 3 if statements corresponding to each part of the homework.
hw = 0 is the lin reg
hw = 1 is grad desc
hw = 2 is rff

Lin reg and grad desc both print out their thetas and losses and the theoretical thetas and losses (from lstsq) for easy comparison.
Rff prints out it's theta, omega, and b.

There are corresponding print and visualize function calls for 1D and 2D input data which are labeled by comments. 