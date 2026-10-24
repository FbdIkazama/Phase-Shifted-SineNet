#ifndef STRUCT_H
#define STRUCT_H

/* Defines whose activation function are available */
typedef enum {
    ACT_SIGMOID,
    ACT_RELU,
    ACT_SINE,
    ACT_TANH,
    ACT_LINEAR,
    ACT_SOFTMAX
} ActivationType;

/* Define the structure of a layer
* @param numNodes Number of nodes for this layer 
* @param inputs Array of pre-activation values 
* @param activations Array of post-activation values 
* @param deltas Array used to store the error during backpropagation 
* @param act Defines which activation function is used on current layer */
typedef struct {
    int numNodes;
    float *inputs;      
    float *activations; 
    float *deltas;
    ActivationType act;
} Layer;

/* Define the general structure of the weights' vector (of a network with N layers)
* Set a lerning rate of 0.001f when no optimization is applied and 0.0001f for RMSProp optimizer. Otherwise use 0.01f for Momentum and Adam optimizers.
* @param inputLayer Defines the previous layer (layer 0 for input layer, 1-2-3 to N-1 for hidden layers)
* @param outputLayer Defines the next layer (1-2-3 to N-1 for hidden layers, N is the output layer)
* @param optimizer Select the optimizer to use (-1 = no optimization, 0 = standard momentum, 1 = RMSProp, 2 = Adam)
* @param learning_rate Learning rate of the network. A decay function is used to contain optimizers spikes
* @param values Stores the current weights as 1D vector (indexing performed with row + offset to increase performances)
* @param phases Stores the phase calculated for each connection
* @param momentum_w Weights' momentum
* @param momentum_phi Phases' momentum
* @param variance_w Weights' variance (RMSProp formula)
* @param variance_phi Phases' momentum (RMSProp formula)*/
typedef struct {
    int inputLayer;
    int outputLayer;
    int optimizer;
    float learning_rate;
    float *values;
    float *phases;
    float *momentum_w;
    float *momentum_phi;
    float *variance_w;
    float *variance_phi;
} Weights;

/* Define the structure of the Sine Net
* @param numLayers Defines the number of layers (including input and output)
* @param layers Array of Layers (number defined by numLayers)
* @param weights Array of matrix weights (number defined by numlayers - 1)
* @param verbose If 0 avoid printing unecessary infos, elsewhere if not 0 prints everything */
typedef struct {
    int numLayers;
    float *loss;
    float *output_val;
    Layer *layers;      
    Weights *weights;
    int verbose;
} SineNet;

#endif // STRUCT_H