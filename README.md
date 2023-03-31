# how_to_build

1. https://github.com/grensen/how_to_build
2. https://github.com/grensen/multi-core
3. https://github.com/grensen/good_vs_bad_code
4. https://github.com/grensen/how_to_train

## Neural Networks
<p align="center">
  <img src="https://github.com/grensen/how_to_build/blob/main/figures/neural_networks.png?raw=true">
</p>

When I think of neural networks, I imagine something similar to the image depicted above. This is the first article in a series of guides that will teach you how to build, train, and test neural networks using a concrete example in C#. In Article 2, we'll modify the neural network to leverage multi-core processors and explore why this leads to non-reproducibility and varying results across multiple training sessions. In Article 3, we will explore both effective and ineffective coding practices, as well as optimization techniques such as SIMD, to achieve the best possible results for our neural network. Article 4 focuses on optimizing the neural network training process and evaluating the performance achieved, with a focus on achieving the best possible results. 

## 105 Miss Predictions
<p align="center">
  <img src="https://github.com/grensen/how_to_build/blob/main/figures/incorrect_105.png?raw=true">
</p>

## Supervised Learning Is Target Learning
<p align="center">
  <img src="https://github.com/grensen/how_to_build/blob/main/figures/naive_learning.png?raw=true">
</p>

## Neurons + Weights = Forward Pass
<p align="center">
  <img src="https://github.com/grensen/how_to_build/blob/main/figures/init_neurons_weights_indices.png?raw=true">
</p>

## Backward Pass = Gradients + Deltas
<p align="center">
  <img src="https://github.com/grensen/how_to_build/blob/main/figures/init_gradients_deltas_indices.png?raw=true">
</p>

## The Process For Every Weight
<p align="center">
  <img src="https://github.com/grensen/how_to_build/blob/main/figures/neural_network_process.png?raw=true">
</p>

## Batch 1
<p align="center">
  <img src="https://github.com/grensen/how_to_build/blob/main/figures/batch1.png?raw=true">
</p>

## Batch 2
<p align="center">
  <img src="https://github.com/grensen/how_to_build/blob/main/figures/batch2.png?raw=true">
</p>

## Weight Update
<p align="center">
  <img src="https://github.com/grensen/how_to_build/blob/main/figures/Update_batch.png?raw=true">
</p>

## New Batch
<p align="center">
  <img src="https://github.com/grensen/how_to_build/blob/main/figures/new_batch.png?raw=true">
</p>

## Activations: Hidden = ReLU, Output = Softmax
<p align="center">
  <img src="https://github.com/grensen/how_to_build/blob/main/figures/neural_network_activations.png?raw=true">
</p>

## Understand ReLU Activations With DESMOS
<p align="center">
  <img src="https://github.com/grensen/how_to_build/blob/main/figures/ReLU_ji.png?raw=true">
</p>

## Understand Softmax Activations 
<p align="center">
  <img src="https://github.com/grensen/how_to_build/blob/main/figures/exp_ji.png?raw=true">
</p>

## Latex
<p align="center">
  <img src="https://github.com/grensen/how_to_build/blob/main/figures/perceptron_vs_layer.png?raw=true">
</p>

## Feed Forward Perceptron-Wise
<p align="center">
  <img src="https://github.com/grensen/how_to_build/blob/main/figures/ff_perceptron-wise.gif?raw=true">
</p>

## Feed Forward Layer-Wise
<p align="center">
  <img src="https://github.com/grensen/how_to_build/blob/main/figures/ff_layer-wise.gif?raw=true">
</p>

## Init Your Picture
<p align="center">
  <img src="https://github.com/grensen/how_to_build/blob/main/figures/init_with_input.png?raw=true">
</p>

## Trained With 50% Pseudo Probability For One Problem
<p align="center">
  <img src="https://github.com/grensen/how_to_build/blob/main/figures/trained_with_input2.png?raw=true">
</p>


## High Level Overview
~~~
1. Preprocess the data.
2. Define the model.
3. Train the model.
4. Evaluate the model.
~~~

## Code Intuition
<p align="center">
  <img src="https://github.com/grensen/how_to_build/blob/main/figures/network_intuition.png?raw=true">
</p>

## High Level Code
~~~cs
// https://github.com/grensen/how_to_build/
#if DEBUG
    System.Console.WriteLine("Debug mode is on, switch to Release mode");
#endif 
System.Action<string> print = System.Console.WriteLine;

print("\nBegin how to build neural networks demo\n");

// get data
AutoData d = new(@"C:\mnist\");

// define neural network 
int[] network       = { 784, 100, 100, 10 };
var LEARNINGRATE    = 0.0005f;
var MOMENTUM        = 0.67f;
var EPOCHS          = 50;
var BATCHSIZE       = 800;
var FACTOR          = 0.99f;

RunDemo(d, network, LEARNINGRATE, MOMENTUM, FACTOR, EPOCHS, BATCHSIZE);

print("\nEnd how to build neural networks demo");

static void RunDemo(AutoData d, int[] NET, float LR, float MOM, float FACTOR, int EPOCHS, int BATCHSIZE){}
static float RunNet(AutoData d, Net neural, int len, float LR, float MOM, float FACTOR, int EPOCHS, int BATCHSIZE){}
static bool EvalAndTrain(int x, byte target, byte[] samples, Net neural, float[] delta){}
static bool EvalTest(int x, byte label, byte[] samples, Net neural){}
Eval(int x, byte[] samples, Net neural, float[] neuron){}
struct Net{}
struct AutoData{}
class Erratic{}
~~~

## DALL-E 2 Neural Networks
<p align="center">
  <img src="https://github.com/grensen/how_to_build/blob/main/figures/DALL_E_neural_networks.png?raw=true">
</p>
