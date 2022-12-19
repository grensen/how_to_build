# how_to_build

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


