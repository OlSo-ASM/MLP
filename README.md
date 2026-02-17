# MLP
A simple neural network for recognizing handwritten digits 
in C++, using SFML and eigen 5.0.0, with a machine learning algorithm 
written using the MNIST database. 

# build
!You need to send unziped "mnist_train.csv" and arial.ttf into folder with exe!
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

# control

left mouse button - drawing
right mouse button - erasing
delete - cleaning the canvas
