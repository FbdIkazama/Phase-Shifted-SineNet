// Hyperparameters
#ifndef LEARNING_RATE
#define LEARNING_RATE 0.01f  // Starting learning rate
#endif
#ifndef LEARNING_DECAY
#define LEARNING_DECAY 0.5f  // LEARNING_RATE scaling factor after DECAY_STEP epochs (for Adam optimizer)
#endif
#ifndef DECAY_STEP
#define DECAY_STEP 500  // Lowering this value produces a more aggressive learning (for Adam optimizer) causing unpredictable results
#endif
#ifndef MOMENTUM_BETA
#define MOMENTUM_BETA 0.9f  // Momentum beta parameter
#endif
#ifndef RMS_BETA
#define RMS_BETA 0.999f  // RMSProp beta parameter
#endif
#ifndef E
#define E 0.00000001f  // Low value to avoid division by 0 during weight optimization
#endif

// Net structure parameters
#ifndef N_LAYERS
#define N_LAYERS 3  // Number of layers in the network
#endif
#ifndef N_NODES_LAYER0
#define N_NODES_LAYER0 4 // Number of nodes in the 1st layer
#endif
#ifndef N_NODES_LAYER1
#define N_NODES_LAYER1 12 // Number of nodes in the 2nd layer
#endif
#ifndef N_NODES_LAYER2
#define N_NODES_LAYER2 3 // Number of nodes in the 3rd layer
#endif
#ifndef N_WEIGHTS
#define N_WEIGHTS 2  // Number of matrix weights
#endif

// Training parameters
#ifndef EPOCHS
#define EPOCHS 2000
#endif

#ifndef PATIENCE 
#define PATIENCE 15
#endif

#ifndef RMSE
#define RMSE 0
#endif

#ifndef CE
#define CE 1
#endif

// Dataset parameters
#define FUNC "Iris_oneHot" 
#define RESOURCES_PATH  "../resources/"