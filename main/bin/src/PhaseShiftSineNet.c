#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "math_utils.h"
#include "settings.h"
#include "data_utils.h"
#include "struct.h"

/* Initializes the values of the structures and allocates memory */
void _init(SineNet *net) {
    // Init the layers values to 0.0f
    for (int l = 0; l < net->numLayers; l++) {
        net->layers[l].inputs = (float *)malloc(net->layers[l].numNodes * sizeof(float));
        net->layers[l].activations = (float *)malloc(net->layers[l].numNodes * sizeof(float));
        net->layers[l].deltas = (float *)malloc(net->layers[l].numNodes * sizeof(float));

        for (int i = 0; i < net->layers[l].numNodes; i++) {
            net->layers[l].inputs[i] = 0.0f;
            net->layers[l].activations[i] = 0.0f;
            net->layers[l].deltas[i] = 0.0f;
        }
    }

    // Allocate memory to store RMSE error value during training
    net->loss = (float*)malloc(EPOCHS * sizeof(float));
    for (int i = 0; i < EPOCHS; i++) {
        net->loss[i] = 0.0f;
    }

    // Init weights parameters
    for (int wl = 0; wl < net->numLayers - 1; wl++) {
        // Weights stored as 1D vector with size nodeIn*nodeOut (assume fully connected)
        int nodeIn = net->layers[wl].numNodes;
        int nodeOut = net->layers[wl+1].numNodes;
        net->weights[wl].values = (float *)malloc(nodeIn * nodeOut * sizeof(float));

        // Initialize phases and momentum at 0.0f
        net->weights[wl].phases = (float *)calloc(nodeIn * nodeOut, sizeof(float));
        net->weights[wl].momentum_w = (float *)calloc(nodeIn * nodeOut, sizeof(float));
        net->weights[wl].momentum_phi = (float *)calloc(nodeIn * nodeOut, sizeof(float));
        net->weights[wl].variance_w = (float *)calloc(nodeIn * nodeOut, sizeof(float));
        net->weights[wl].variance_phi = (float *)calloc(nodeIn * nodeOut, sizeof(float));

        // Handle Adam optimization
        net->weights[wl].optimizer = 2;
        net->weights[wl].learning_rate = LEARNING_RATE;

        // SIREN initialization has been used: U(-1/nodeIn, 1/nodeIn)
        if (wl == 0) { // First layer -> SIREN scaled for frequency omega = 30
            float limit = 30.0f / (float)nodeIn; 
            
            for (int i = 0; i < nodeIn * nodeOut; i++) { 
                float rnd_base = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
                net->weights[wl].values[i] = rnd_base * limit;
            }
        } else { // Second layer -> SIREN initialization U(-6/nodeIn, 6/nodeIn)
            float limit = sqrt(6.0f / (float)nodeIn);
            for (int i = 0; i < nodeIn * nodeOut; i++) { 
                float rnd_base = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
                net->weights[wl].values[i] = rnd_base * limit;
            }
        }
    }
}

/* Updates the weights between two consecutive layers
* @param prevLayer Previous layer structure
* @param w Structure of weights that connects previous with current layer
* @param currLayer Current layer structure */
void _weightsAndPhasesUpdate(Layer prevLayer, Weights *w, Layer currLayer) {
    float scaleFactor = (2.0f * M_PI) / prevLayer.numNodes;

    #pragma omp parallel for
    for (int i = 0; i < currLayer.numNodes; i++) {
        for (int j = 0; j < prevLayer.numNodes; j++) {
            int idx = i * prevLayer.numNodes + j;

            float x = prevLayer.activations[j];
            float weight = w->values[idx];
            float phase = w->phases[idx];
            
            float phase_grad = cos(weight * x + phase);
            float grad = currLayer.deltas[i] * scaleFactor * phase_grad;
            float grad_w = grad * x;
            float grad_phi = grad;
            
                    

            switch (w->optimizer) {
            case 0:
                /* Standard momentum optimizer 
                * new_momentum = prev_momentum * momentum_beta + (1 - momentum_beta) * gradient 
                * new_weight/phase = prev_weight/phase * learning_rate */
                w->momentum_w[idx] = (w->momentum_w[idx] * MOMENTUM_BETA) + ((1.0f - MOMENTUM_BETA) * grad_w);
                w->momentum_phi[idx] = (w->momentum_phi[idx] * MOMENTUM_BETA) + ((1.0f - MOMENTUM_BETA) * grad_phi);
                
                w->values[idx] += w->momentum_w[idx] * LEARNING_RATE;
                w->phases[idx] += w->momentum_phi[idx] * LEARNING_RATE;
                break;
            case 1:
                /* RMSProp optimizer
                * new_variance = prev_variance * RMSbeta + (1 - RMSbeta) * gradient^2 
                * new_weight/phase = learning_rate / sqrt(prev_variance + epsilon) * gradient */
                w->variance_w[idx] = (w->variance_w[idx] * RMS_BETA) + ((1.0f - RMS_BETA) * grad_w * grad_w);
                w->variance_phi[idx] = (w->variance_phi[idx] * RMS_BETA) + ((1.0f - RMS_BETA) * grad_phi * grad_phi);
                
                w->values[idx] += (LEARNING_RATE / sqrt(w->variance_w[idx] + E)) * grad_w;
                w->phases[idx] += (LEARNING_RATE / sqrt(w->variance_phi[idx] + E)) * grad_phi;
                break;
            case 2:
                /* Adam optimizer
                * new_momentum = prev_momentum * momentum_beta + (1 - momentum_beta) * gradient
                * new_variance = prev_variance * RMSbeta + (1 - RMSbeta) * gradient^2 
                * new_weight/phase = learning_rate * prev_momentum / sqrt(prev_variance + epsilon) */
                w->momentum_w[idx] = (w->momentum_w[idx] * MOMENTUM_BETA) + ((1.0f - MOMENTUM_BETA) * grad_w);
                w->momentum_phi[idx] = (w->momentum_phi[idx] * MOMENTUM_BETA) + ((1.0f - MOMENTUM_BETA) * grad_phi);

                w->variance_w[idx] = (w->variance_w[idx] * RMS_BETA) + ((1.0f - RMS_BETA) * grad_w * grad_w);
                w->variance_phi[idx] = (w->variance_phi[idx] * RMS_BETA) + ((1.0f - RMS_BETA) * grad_phi * grad_phi);

                w->values[idx] += w->learning_rate * w->momentum_w[idx] / (sqrt(w->variance_w[idx]) + E);
                w->phases[idx] += w->learning_rate * w->momentum_phi[idx] / (sqrt(w->variance_phi[idx]) + E);
                break;
            default:
                // No optimization
                w->values[idx] += grad_w * LEARNING_RATE;
                w->phases[idx] += grad_phi * LEARNING_RATE;
                break;
            }
        }
    }
}

/* Calculates the node activation 
* @param layer The layer with pre-activated nodes
* @param type The function activation used for the current layer. The matching value it store within ActivationType enum */
void _activation(Layer layer, ActivationType type) {
    if (type == ACT_SOFTMAX) {
        _softmax(layer);
        return;
    }

    for (int node = 0; node < layer.numNodes; node++) {
        float val = layer.inputs[node];
        if (type == ACT_SIGMOID) _sigmoid(&val);
        else if (type == ACT_RELU) _relu(&val);
        else if (type == ACT_TANH) _tanh(&val);
        layer.activations[node] = val;
    }
}

/* Calculates the node value using the Sine Net forward pass formula
* I(s) = 2*PI/input*(sum from 0 to N of (sin(w*x))) | Calculates sin for each weighted input and then sums for each node on the previous layer.
* Where: s is the current layer, N is the number of nodes in the previous layer, w are the weights from the previous to the current layer,
* x is the activation value of the node in the previous layer, input is the number of nodes as input 
* @param prevLayer Previous layer structure
* @param w Structure of weights that connects previous with current layer
* @param currLayer Current layer structure */
void _calcNodeValue(Layer prevLayer, Weights w, Layer currLayer) {
    #pragma omp parallel for
    for (int currNode = 0; currNode < currLayer.numNodes; currNode++) {
        float sum = 0.0f;
        for (int prevNode = 0; prevNode < prevLayer.numNodes; prevNode++) {
            int idx = currNode * prevLayer.numNodes + prevNode;
            sum += sin(w.values[idx] * prevLayer.activations[prevNode] + w.phases[idx]);
        }
        currLayer.inputs[currNode] = sum * (2.0*M_PI)/prevLayer.numNodes;
    }
    _activation(currLayer, currLayer.act);
}

/* Calculates the error of each node in the output layer as: 
* f'(I(out))*(target-x(out)) | Where f'(I(out)) is the first derivative of the activation function of the output layer,
* x(out) is the value post-activation of the output node
* @param outputLayer Output layer structure of the network
* @param target Supervised learning target */
void _calcOutputError(Layer outputLayer, float *target) {
    float val, deriv;
    for (int node = 0; node < outputLayer.numNodes; node++) {
        val = outputLayer.activations[node];
        
        if (outputLayer.act == ACT_TANH) deriv = (1.0f - val * val);
        else if (outputLayer.act == ACT_SIGMOID) deriv = val * (1.0f - val);
        else if (outputLayer.act == ACT_RELU) deriv = (val > 0) ? 1.0f : 0.0f;
        else deriv = 1.0f;

        outputLayer.deltas[node] = deriv * (target[node] - val);
    }
}

/* Calculates the error of each node in each layer using backpropagation as: 
* f'(I(s))*(2*PI/numNodes) * sum from 0 to k (e(s+1) * w(s+1) * cos(w(s+1)*x))| Where f'(I(s)) is the first derivative of the activation function of the s-layer,
* k is the current node
* numNodes is the number of the input nodes
* e(s+1) is the error in the next layer of the k-th node
* w(s+1) is the weight connecting nodes from current to next layer
* x is the activation value of the node
* @param currLayer Current layer
* @param w Vector of weights
* @param nextLayer Next layer
*/
void _calcLayerError(Layer currLayer, Weights w, Layer nextLayer) {
    #pragma omp parallel for
    for (int i = 0; i < currLayer.numNodes; i++) {
        float errorSum = 0.0f;
        for (int j = 0; j < nextLayer.numNodes; j++) {
            int idx = j * currLayer.numNodes + i;
            errorSum += nextLayer.deltas[j] * w.values[idx] * cos(w.values[idx] * currLayer.activations[i] + w.phases[idx]);
        }
        float val = currLayer.activations[i];
        float deriv;

        if (currLayer.act == ACT_TANH) deriv = (1.0f - val * val);
        else if (currLayer.act == ACT_SIGMOID) deriv = val * (1.0f - val);
        else if (currLayer.act == ACT_RELU) deriv = (val > 0) ? 1.0f : 0.0f;
        else deriv = 1.0f;

        currLayer.deltas[i] = errorSum * deriv;
    }
}

/* Manage validation algorithm */
void validate(SineNet *net, float **dataset, int idx, int val_size, float **target, float *error_val, int dataset_size) {
    for (int record = idx; record < idx + val_size; record++) {

        // Updates the first layer with input values
        memcpy(net->layers[0].activations, dataset[record], net->layers[0].numNodes * sizeof(float));

        // Forward pass -> calculates node values and activation for each layer (from the first hidden layer)
        for (int i = 1; i < net->numLayers; i++) {
            _calcNodeValue(net->layers[i-1], net->weights[i-1], net->layers[i]);
        }

        for (int i = 0; i < net->layers[net->numLayers-1].numNodes; i++) {
            if (RMSE) {
                float diff = target[record][i] - net->layers[net->numLayers-1].activations[i];
                (*error_val) += diff * diff;
            }

            if (CE && target[record][i] > 0.0f) { 
                (*error_val) -= target[record][i] * logf(net->layers[net->numLayers-1].activations[i] + E);
            }
        }
    }
}

/* Training algorithm of the SineNet based on a classic Stochastic Gradient Descent
* @param *net Memory address of the network to be trained
* @param **dataset Matrix storing the whole dataset 
* @param dataset_size Number of rows of the dataset matrix 
* @param batch_size Number of record for each batch 
* @param **target Matrix containing the target for each record */
void train(SineNet *net, float **dataset, int dataset_size, int batch_size, float **target) {
    int train_size = (int)(dataset_size * 0.8); // Extract the validation set from the dataset (80% training, 20% validation)
    int val_size = dataset_size - train_size;

    int wait = 0;                    // Counter of epochs without loss reduction
    float best_val_loss = 999999.0f; // Stores the minimum loss value

    Weights w_cpy[N_LAYERS-1];

    for (int i = 0; i < net->numLayers - 1; i++) {
        int size = net->layers[i].numNodes * net->layers[i+1].numNodes;
        w_cpy[i].values = (float *)malloc(size * sizeof(float));
        w_cpy[i].phases = (float *)malloc(size * sizeof(float));
    }

    printf("\nStart training...\n");
    if (!net->verbose) printf("\033[?25l"); // Hide terminal cursor (to hide cursor artifacts)

    // Training on the other part of the dataset
    for (int currentEpoch = 0; currentEpoch < EPOCHS; currentEpoch++) {
        float squared_error = 0;
        for (int batch = 0; batch < train_size/batch_size; batch++) {
            for (int record = 0; record < batch_size; record++) {
                int idx = batch * batch_size + record;

                // Updates the first layer with input values
                memcpy(net->layers[0].activations, dataset[idx], net->layers[0].numNodes * sizeof(float));

                // Forward pass -> calculates node values and activation for each layer (from the first hidden layer)
                for (int i = 1; i < net->numLayers; i++) {
                    _calcNodeValue(net->layers[i-1], net->weights[i-1], net->layers[i]);
                }

                // Output error calculation
                _calcOutputError(net->layers[net->numLayers-1], target[idx]);

                // Calculates the squared error after each output calculation
                for (int i = 0; i < net->layers[net->numLayers-1].numNodes; i++) {
                    float diff = target[idx][i] - net->layers[net->numLayers-1].activations[i];
                    squared_error += diff * diff;
                }

                /* Calculation of errors for each node in each layer,
                * followed by backpropagation weight update.*/
                for (int i = net->numLayers - 2; i >= 0; i--) {
                    _calcLayerError(net->layers[i], net->weights[i], net->layers[i+1]);
                    _weightsAndPhasesUpdate(net->layers[i], &net->weights[i], net->layers[i+1]);
                }
            }
        }
        // Calculates the RMSE training loss after each epoch
        float training_rmse = sqrt(squared_error / (train_size / batch_size) * batch_size);

        // Calculates the validation loss after each epoch
        float error_val = 0;
        validate(net, dataset, train_size, val_size, target, &error_val, dataset_size);

        float validation_err;
        if (RMSE) validation_err = sqrt(error_val / val_size);  // RMSE loss function
        else if (CE) validation_err = error_val / val_size;     // Cross Entropy loss function

        net->loss[currentEpoch] = validation_err;

        if (validation_err < best_val_loss) {
            best_val_loss = validation_err;
            wait = 0;

            // Saves weights and phases after each loss reduction
            for (int i = 0; i < net->numLayers - 1; i++) {
                int size = net->layers[i].numNodes * net->layers[i+1].numNodes;
                memcpy(w_cpy[i].values, net->weights[i].values, size * sizeof(float));
                memcpy(w_cpy[i].phases, net->weights[i].phases, size * sizeof(float));
            }
        } else  {
            wait++;

            if (wait > PATIENCE) {
                if (net->verbose) {
                    printf("\n-------------------------------\n");
                    printf("EARLY STOPPING triggered at epoch %d!\n", currentEpoch + 1);
                    printf("Best Validation Loss: %f reached %d epochs ago.\n", best_val_loss, PATIENCE);
                    printf("-------------------------------\n");
                } else {printf("\nEARLY STOPPING triggered at epoch %d!", currentEpoch + 1);}

                // Restores weights and phases after triggering early stopping 
                for (int i = 0; i < net->numLayers - 1; i++) {
                    int size = net->layers[i].numNodes * net->layers[i+1].numNodes;
                    memcpy(net->weights[i].values, w_cpy[i].values, size * sizeof(float));
                    memcpy(net->weights[i].phases, w_cpy[i].phases, size * sizeof(float));
                }
                break;
            }
        }
        
        // Handle Adam optimizer learning decay -> smoother RMSE decay during training
        if (net->weights->optimizer == 2 && (currentEpoch + 1) % DECAY_STEP == 0) {
            for (int i = 0; i < net->numLayers-1; i++) {
                net->weights[i].learning_rate *= LEARNING_DECAY;
            }
        }

        if (net->verbose) {
            printf("-------------------------------\n");
            printf("Epoch %d completed\n", currentEpoch+1);
            printf("Training loss - RMSE: %f \n", training_rmse);
            printf("Validation loss - RMSE: %f \n", validation_err);
            printf("-------------------------------");
        } else {
            float progress = (float)(currentEpoch + 1) / (float)EPOCHS * 100.0f;
            printf("\rProgress: ");
            int barWidth = 20;
            int numHashes = (int)(progress / (100.0f / barWidth));
            for (int i = 0; i < barWidth; i++) {
                if (i < numHashes) printf("#");
                else printf("-");
            }
            printf(" %1.0f%%", progress);
            fflush(stdout);
        }
    }
    if (!net->verbose) printf("\033[?25h\n"); // Restore terminal cursor

    // Free backup weights memory allocated for deep copies
    for (int i = 0; i < net->numLayers - 1; i++) {
        free(w_cpy[i].values);
        free(w_cpy[i].phases);
    }
}

int main(int argc, char *argv[]) {
     if (argc < 7) return -1;
    
    char *dataset_path = argv[1];
    char *loss_path = argv[2];
    char *out_path = argv[3];
    char *out_roc_path = argv[4];
    int opt = atoi(argv[5]);
    int verb = atoi(argv[6]);
    //float learning_rate = atof(argv[6]);
    //int epochs = atoi(argv[7]);


    srand(1); // Dynamic seed based on the run
    int dataset_size;
    float **dataset, **targets;
    float loss_epochs[EPOCHS];
    float input_mins[N_NODES_LAYER0];
    float input_maxs[N_NODES_LAYER0];
    float target_mins[N_NODES_LAYER2];
    float target_maxs[N_NODES_LAYER2];

    Layer inLayer, hidden1, outLayer;
    inLayer.numNodes = N_NODES_LAYER0;
    inLayer.act = ACT_TANH;
    hidden1.numNodes = N_NODES_LAYER1;
    hidden1.act = ACT_TANH;
    outLayer.numNodes = N_NODES_LAYER2;
    outLayer.act = ACT_SOFTMAX;
    Layer layers[N_LAYERS] = {inLayer, hidden1, outLayer};

    Weights w1, w2;
    w1.inputLayer = 0; w1.outputLayer = 1;
    w2.inputLayer = 1; w2.outputLayer = 2;
    Weights w[N_WEIGHTS] = {w1, w2};
    
    SineNet sineNet;
    sineNet.numLayers = N_LAYERS;
    sineNet.loss = loss_epochs;
    sineNet.layers = layers;
    sineNet.weights = w;
    for (int i = 0; i < N_WEIGHTS; i++) {
        sineNet.weights[i].optimizer = opt;
    }
    sineNet.verbose = verb;

/*     char dataset_path[256], loss_path[256], out_path[256];
    snprintf(dataset_path, sizeof(dataset_path), "%s%s%s", RESOURCES_PATH, FUNC, " - dataset.csv");
    snprintf(loss_path, sizeof(loss_path), "%s%s%s", RESOURCES_PATH, FUNC, " - loss.csv");
    snprintf(out_path, sizeof(out_path), "%s%s%s", RESOURCES_PATH, FUNC, " - curve.csv");  */

    //_generate_func_dataset(dataset_path, -3*M_PI, 3*M_PI);

    _init(&sineNet);

    _get_csv_dim(dataset_path, &dataset_size);
    if (dataset_size > 0) dataset_size--; // Exclude the header line

    _init_data(&dataset, &targets, dataset_size, N_NODES_LAYER0, N_NODES_LAYER2);

    _read_csv(dataset_path, dataset, targets, dataset_size, N_NODES_LAYER0, N_NODES_LAYER2);

    _data_scaling(dataset, 0, N_NODES_LAYER0, dataset_size, -1.0f, 1.0f, input_mins, input_maxs);
    
    //_data_scaling(targets, 0, N_NODES_LAYER2, dataset_size, -1.0f, 1.0f, target_mins, target_maxs);

    train(&sineNet, dataset, dataset_size, 10, targets);

    _dump_loss(sineNet.loss, loss_path);

    //_export_XOR_decision_boundary(sineNet, "XOR_heatmap.csv", _calcNodeValue);
    //_export_curve(sineNet, curve_path, _calcNodeValue, input_mins, input_maxs, target_mins, target_maxs);
    _export_confusion_matrix(sineNet, dataset_size, dataset, targets, out_path, out_roc_path, _calcNodeValue);

    _free_memory(dataset, targets, &sineNet, dataset_size);

    return 0;
}