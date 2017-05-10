# Meetup Barranquilla: Reinforcement learning

Deep Q-learning agent for replicating DeepMind's results in paper ["Human-level control through deep reinforcement learning"](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html). It is designed to be simple, fast and easy to extend.
<img src="https://keon.io/images/deep-q-learning/rl.png" width="450">

## Overview
This project follows the description of the Deep Q Learning algorithm described in Playing Atari with Deep Reinforcement Learning [2] and shows that this learning algorithm can be further generalized to the notorious Flappy Bird, Pong and Space Invaders.

## Installation
1) Install Tensorflow: 
`pip install tensorflow` 
If you have NVIDIA GPU, you should do:
`pip install tensorflow-gpu`
2) Install pygame:
`pip install pygame`
3) Install opencv:
`pip install opencv-python`
4) Install all gym dependencies:
`pip install gym'[all]'`

**Another way:**
`pip install -r requirements.txt`


## What is Deep Q-Network?
It is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards. For those who are interested in deep reinforcement learning, I highly recommend to read the following post:
[Demystifying Deep Reinforcement Learning](http://www.nervanasys.com/demystifying-deep-reinforcement-learning/)

<img src="https://www.nervanasys.com/wp-content/uploads/2016/04/Screen-Shot-2016-04-27-at-10.59.50-AM.png" width="450">

## FlappyBird Architecture 

According to [1], I first preprocessed the game screens with following steps:

1. Convert image to grayscale
2. Resize image to 80x80
3. Stack last 4 frames to produce an 80x80x4 input array for network

**DQN: Network**
1. First layer convolves the input image with an 8x8x4x32 kernel at a stride size of 4.
2. The output is then put through a 2x2 max pooling layer. 
3. The second layer convolves with a 4x4x32x64 kernel at a stride of 2. 
4. The third layer convolves with a 3x3x64x64 kernel at a stride of 1.
5. FullyConnected 1600x512
6. Readout: FullyConnected 512x2


** Training **
For training, you should do the following:

  ```
  cd FlappyBird
  python deep_q_network.py
  ```
## Examples: GYM 
There are examples developed in OpenAI gym for the pong game and SpaceInvaders.
Just:

  ```
  cd atari
  python pong.py
  python SpaceInvaders.py 
  ```

## References

[1] Mnih Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. **Human-level Control through Deep Reinforcement Learning**. Nature, 529-33, 2015.

[2] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. **Playing Atari with Deep Reinforcement Learning**. NIPS, Deep Learning workshop

[3] Kevin Chen. **Deep Reinforcement Learning for Flappy Bird** [Report](http://cs229.stanford.edu/proj2015/362_report.pdf) | [Youtube result](https://youtu.be/9WKBzTUsPKc)

## Disclaimer
This work is highly based on the following repos:

1. [sourabhv/FlapPyBird](https://github.com/sourabhv/FlapPyBird)
2. [asrivat1/DeepLearningVideoGames](https://github.com/asrivat1/DeepLearningVideoGames)

## Slides
[Google Docs] https://goo.gl/9UJNCu
