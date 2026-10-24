#ifndef MATH_UTILS_H
#define MATH_UTILS_H
#include <math.h>
#include "settings.h"
#include "struct.h"

/*************** Activation functions ***************/

/* Sigmoid activation function
* @param x Node value to be activated */
void _sigmoid(float *x) {
    *x = 1.0f / (1.0f + expf(-(*x)));
}

/* ReLU activation function
* @param x Node value to be activated */
void _relu(float *x) {
    *x = fmax(0.0, *x);
}

/* Tanh activation function
* @param x Node value to be activated */
void _tanh(float *x) {
    *x = tanhf(*x);
}

/* Linear activation function
* @param x Node value to be activated */
void _linear(float *x) {
    *x = *x;
}

/* Softmax activation function 
* @param x Value to be activated
* @param outputs Node values of output layer */
void _softmax(Layer layer) {
    float max_val = layer.inputs[0];
    for (int i = 0; i < layer.numNodes; i++) {
        if (layer.inputs[i] > max_val) max_val = layer.inputs[i];
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < layer.numNodes; i++) {
        layer.activations[i] = expf(layer.inputs[i] - max_val);
        sum_exp += layer.activations[i];
    }

    for (int i = 0; i < layer.numNodes; i++) {
        layer.activations[i] /= sum_exp;
    }
}

/*************** Scaling functions ***************/
/* Implements a simple min-max scaler function */
void _min_max_scaler_col(float **dataset, int col, int dataset_size, float target_min, float target_max, float *data_mins, float *data_maxs) {
    float c_min = dataset[0][col];
    float c_max = dataset[0][col];

    for (int i = 1; i < dataset_size; i++) {
        if (dataset[i][col] > c_max) c_max = dataset[i][col];
        if (dataset[i][col] < c_min) c_min = dataset[i][col];
    }

    data_mins[col] = c_min;
    data_maxs[col] = c_max;

    if (c_max - c_min == 0.0f) return;

    for (int i = 0; i < dataset_size; i++) {
        float x = dataset[i][col];
        dataset[i][col] = target_min + ((x - c_min) * (target_max - target_min)) / (c_max - c_min);
    }
}

/* Apply the reverse scaling to the data for correct comparison */
void _inverse_scaler(float *x, float target_min, float target_max, float data_min, float data_max) {
    *x = (*x - target_min) * (data_max - data_min)/(target_max - target_min) + data_min;
}

#endif // MATH_UTILS_H