#pragma once

#include "nn/core/tensor.h"

void init_xavier_uniform(Tensor* tensor);
void init_xavier_normal(Tensor* tensor);
void init_kaiming_uniform(Tensor* tensor);
void init_kaiming_normal(Tensor* tensor);