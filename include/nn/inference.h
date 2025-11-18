#pragma once

#include "nn/cnn.h"

CNN* cnn_load_model(int epoch);
void cnn_free_model(CNN* model);