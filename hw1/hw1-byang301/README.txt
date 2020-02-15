Bojun Yang

My code is tested in the main method.
I import the data in the beginning then I have commented out code for 
duplicating features. I also append the vector of 1s before I call any function.
I also append a vector of 1s in apply_RFF_transform. 
My reasoning is stated in my follow up discussings for this piazza question.
https://piazza.com/class/k4x9jo9fhkz18r?cid=17

After appending 1s, I calculate the theoretical data with lstsq.
I have 3 if statements corresponding to each part of the homework.
hw = 0 is the lin reg
hw = 1 is grad desc
hw = 2 is rff

Lin reg and grad desc both print out their thetas and losses and the theoretical thetas and losses (from lstsq) for easy comparison.
Rff prints out it's theta, omega, and b.

There are corresponding print and visualize function calls for 1D and 2D input data which are labeled by comments. 

My conda environment is defined below. I just copy pasted the yml file.

name: ml
channels:
  - anaconda
  - defaults
  - conda-forge
dependencies:
  - astroid=2.3.3=py38_0
  - blas=1.0=mkl
  - ca-certificates=2019.11.27=0
  - certifi=2019.11.28=py38_0
  - cycler=0.10.0=py_2
  - freetype=2.10.0=h24853df_1
  - intel-openmp=2019.5=281
  - isort=4.3.21=py38_0
  - kiwisolver=1.1.0=py38ha1b3eb9_0
  - lazy-object-proxy=1.4.3=py38h1de35cc_0
  - libcxx=9.0.1=1
  - libcxxabi=4.0.1=hcfea43d_1
  - libedit=3.1.20181209=hb402a30_0
  - libffi=3.2.1=h475c297_4
  - libgfortran=3.0.1=h93005f0_2
  - libpng=1.6.37=h2573ce8_0
  - matplotlib=3.1.2=py38_1
  - matplotlib-base=3.1.2=py38h11da6c2_1
  - mccabe=0.6.1=py38_1
  - mkl=2019.5=281
  - mkl-service=2.3.0=py38hfbe908c_0
  - mkl_fft=1.0.15=py38h5e564d8_0
  - mkl_random=1.1.0=py38h6440ff4_0
  - ncurses=6.1=h0a44026_1
  - numpy=1.18.1=py38h7241aed_0
  - numpy-base=1.18.1=py38h6575580_0
  - openssl=1.1.1d=h1de35cc_3
  - pip=19.3.1=py38_0
  - pylint=2.4.4=py38_0
  - pyparsing=2.4.6=py_0
  - python=3.8.1=h359304d_1
  - python-dateutil=2.8.1=py_0
  - readline=7.0=h1de35cc_5
  - setuptools=44.0.0=py38_0
  - six=1.13.0=py38_0
  - sqlite=3.30.1=ha441bb4_0
  - tk=8.6.8=ha441bb4_0
  - tornado=6.0.3=py38h0b31af3_0
  - wheel=0.33.6=py38_0
  - wrapt=1.11.2=py38h1de35cc_0
  - xz=5.2.4=h1de35cc_4
  - zlib=1.2.11=h1de35cc_3

