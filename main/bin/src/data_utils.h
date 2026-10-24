#include <stdio.h>
#include "settings.h"
#include "struct.h"
#include "math_utils.h"

// Implementation to parse CSV and fill the allocated arrays
void _read_csv(const char* filename, float **dataset, float **target, int dataset_size, int num_features, int num_targets) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("Error opening file for reading");
        return;
    }

    // Skip header line with non-numeric values
    char buffer[1024];
    if (fgets(buffer, sizeof(buffer), fp) == NULL) {
        fclose(fp);
        return;
    }
    
    for (int i = 0; i < dataset_size; i++) {
        for (int j = 0; j < num_features; j++) {
            if (fscanf(fp, "%f,", &dataset[i][j]) <= 0) break;
        }
        for (int k = 0; k < num_targets; k++) {
            if (k == num_targets - 1) {
                if (fscanf(fp, "%f", &target[i][k]) <= 0) break;
            } else {
                if (fscanf(fp, "%f,", &target[i][k]) <= 0) break;
            }
        }
    }

    fclose(fp);
}

// Implementation to count rows
void _get_csv_dim(const char* filename, int *dataset_size) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("Error opening file for dimension check");
        *dataset_size = 0;
        return;
    }
    
    *dataset_size = 0;
    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
        (*dataset_size)++;
    }

    fclose(fp);
}

void _init_data(float ***dataset, float ***targets, int dataset_size, int num_features, int num_targets) {
    *dataset = (float**)malloc(dataset_size * sizeof(float*));
    *targets = (float**)malloc(dataset_size * sizeof(float*));
    for (int i = 0; i < dataset_size; i++) {
        (*dataset)[i] = (float*)calloc(num_features, sizeof(float));
    }
    for (int i = 0; i < dataset_size; i++) {
        (*targets)[i] = (float*)calloc(num_targets, sizeof(float));
    }
}

void _data_scaling(float **dataset, int start_col, int end_col, int dataset_size, float target_min, float target_max, float *data_mins, float *data_maxs) {
    for (int j = start_col; j < end_col; j++) {
        _min_max_scaler_col(dataset, j, dataset_size, target_min, target_max, data_mins, data_maxs);
    }
}

void _free_memory(float **dataset, float **targets, SineNet *net, int dataset_size) {
    if (dataset != NULL) {
        for (int i = 0; i < dataset_size; i++) {
            if (dataset[i] != NULL) free(dataset[i]);
        }
        free(dataset);
    }
    
    if (targets != NULL) {
        for (int i = 0; i < dataset_size; i++) {
            if (targets[i] != NULL) free(targets[i]);
        }
        free(targets);
    }

    if (net->layers != NULL) {
        for (int l = 0; l < net->numLayers; l++) {
            if (net->layers[l].inputs != NULL) free(net->layers[l].inputs);
            if (net->layers[l].activations != NULL) free(net->layers[l].activations);
            if (net->layers[l].deltas != NULL) free(net->layers[l].deltas);
        }
    }

    if (net->weights != NULL) {
        for (int wl = 0; wl < net->numLayers - 1; wl++) {
            if (net->weights[wl].values != NULL) free(net->weights[wl].values);
            if (net->weights[wl].phases != NULL) free(net->weights[wl].phases);
            if (net->weights[wl].momentum_w != NULL) free(net->weights[wl].momentum_w);
            if (net->weights[wl].momentum_phi != NULL) free(net->weights[wl].momentum_phi);
            if (net->weights[wl].variance_w != NULL) free(net->weights[wl].variance_w);
            if (net->weights[wl].variance_phi != NULL) free(net->weights[wl].variance_phi);
        }
    }

    if (net->loss != NULL) {free(net->loss);}
}

void _dump_loss(float *loss, const char* filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error while saving loss to a file\n");
        return;
    }

    for (int i = 0; i < EPOCHS; i++) {
        if (loss[i] < E) break;
        fprintf(fp, "%f\n", loss[i]);
    }

    fclose(fp);
    printf("\nLoss values exported to %s", filename);
}

/* Creates an high density map used to evaluate the network
* The boundaries are regions where the network can't distinguish the output classes
* The points of the map are generated as result of forward pass throught the network
* @param net The network to be evaluated
* @param filename The file where the map is stored (highly recommended .csv or .txt formats) */
void _export_XOR_decision_boundary(SineNet net, const char* filename, void (*calcFunc)(Layer, Weights, Layer)){
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) return;

    fprintf(fp, "x,y,z\n");

    float step = 0.05f;
    for (float x = -3.0f; x <= 3.0f; x += step) {
        for (float y = -1.5f; y <= 1.5f; y += step) {

            net.layers[0].activations[0] = 1.0f; // Fixed bias
            net.layers[0].activations[1] = x;
            net.layers[0].activations[2] = y;

            for (int i = 1; i < net.numLayers; i++) {
                calcFunc(net.layers[i-1], net.weights[i-1], net.layers[i]);
            }

            float out = net.layers[net.numLayers-1].activations[0];
            fprintf(fp, "%f,%f,%f\n", x, y, out);
        }
    }
    fclose(fp);
    printf("\nDecision boundary exported to %s\n", filename);
}

void _export_curve(SineNet net, const char* filename, void (*calcFunc)(Layer, Weights, Layer), float *input_mins, float *input_maxs, float *target_mins, float *target_maxs) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) return;

    fprintf(fp, "x_real,target_real,pred_real\n");  
   
    for (float x_scaled = -2.0f; x_scaled <= 2.0f; x_scaled += 0.01f) {
        
        net.layers[0].activations[0] = x_scaled;

        for (int i = 1; i < net.numLayers; i++) {
            calcFunc(net.layers[i-1], net.weights[i-1], net.layers[i]);
        }

        float pred_y_scaled = net.layers[net.numLayers-1].activations[0];
        float x_real = x_scaled;
        _inverse_scaler(&x_real, -1.0f, 1.0f, input_mins[0], input_maxs[0]);
        
        float pred_y_real = pred_y_scaled;
        _inverse_scaler(&pred_y_real, -1.0f, 1.0f, target_mins[0], target_maxs[0]);

        float target_y_real = sin(2*x_real) + cos(3*x_real);

        fprintf(fp, "%f,%f,%f\n", x_real, target_y_real, pred_y_real);
    }
    fclose(fp);
    printf("\nReal/predicted function curve exported to %s\n", filename);
}

void _export_confusion_matrix(SineNet net, int dataset_size, float **dataset, float **target, const char* out_path, const char* out_roc_path,void (*calcFunc)(Layer, Weights, Layer)) {
    int **matrix = malloc(N_NODES_LAYER2 * sizeof(int*));
    for (int i = 0; i < N_NODES_LAYER2; i++) {
        matrix[i] = calloc(N_NODES_LAYER2, sizeof(int));
    }

    FILE *fp_roc = fopen(out_roc_path, "w");
    if (fp_roc == NULL) {
        fprintf(stderr, "Error while opening roc output file");
        return;
    }
    fprintf(fp_roc, "y_true,y_prob_malignant\n");

    int real_idx = 0;
    int pred_idx = 0;
    for (int i = (int)(0.8 * dataset_size); i < dataset_size; i++) {
        // Feed the input
        for (int j = 0; j < N_NODES_LAYER0; j++) {
            net.layers[0].activations[j] = dataset[i][j];
        }

        // Forward pass
        for (int l = 1; l < net.numLayers; l++) {
            calcFunc(net.layers[l-1], net.weights[l-1], net.layers[l]);
        }

        // Find the input target
        for (int j = 0; j < N_NODES_LAYER2; j++) {
            if (target[i][j] == 1.0f) {
                real_idx = j;
                break;
            }
        }

        // Find which output node activates the most
        float pred = net.layers[net.numLayers-1].activations[0];
        pred_idx = 0;
        for (int j = 1; j < N_NODES_LAYER2; j++) {
            if (net.layers[net.numLayers-1].activations[j] > pred) {
                pred = net.layers[net.numLayers-1].activations[j];
                pred_idx = j;
            }
        }
        matrix[real_idx][pred_idx]++;

        float prob_malignant = net.layers[net.numLayers-1].activations[1];  // Address the correct output node as present in dataset
        fprintf(fp_roc, "%d,%f\n", real_idx, prob_malignant);
    }
    fclose(fp_roc);

    FILE *fp = fopen(out_path, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error while opening output file");
        return;
    }

    for (int i = 0; i < N_NODES_LAYER2; i++) {
        if (i < N_NODES_LAYER2 - 1) fprintf(fp, "%s %d,", "Pred Class", i);
        else fprintf(fp, "%s %d\n", "Pred Class", i);
    }
    
    for (int i = 0; i < N_NODES_LAYER2; i++) {
        for (int j = 0; j < N_NODES_LAYER2; j++) {
            if (j == N_NODES_LAYER2-1) fprintf(fp, "%d\n", matrix[i][j]);
            else fprintf(fp, "%d,", matrix[i][j]);
        }
    }

    fclose(fp);

    for (int i = 0; i < N_NODES_LAYER2; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void _generate_func_dataset(const char* dataset, float min, float max) {
    FILE *fp = fopen(dataset, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error while opening dataset file\n");
        return;
    }

    srand((unsigned int)time(NULL));

    fprintf(fp, "X,Y\n");
    for (int i = 0; i < 8000; i++) {    // Training set (80%)
        float x = min + ((float)rand() / (float)RAND_MAX) * (max - min);
        fprintf(fp, "%f,%f\n", x, sin(2.0f*x) + cos(3.0f*x));
    }
    for (int i = 0; i < 2000; i++) {    // Validation set -> 20% over the max dataset value
        float x = max + ((float)rand() / (float)RAND_MAX) * (max - min) * 0.2f;
        fprintf(fp, "%f,%f\n", x, sin(2.0f*x) + cos(3.0f*x));
    }
    fclose(fp);
    printf("Dataset correctly generated!\n");
}