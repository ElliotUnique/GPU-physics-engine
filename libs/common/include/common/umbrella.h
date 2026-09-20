#pragma once
// Visible to all
#include "constants.h"
#include "macros.h"
#include "types.h"
// Inisible to nvcc
#ifndef __CUDACC__
#include "thread-control.h"
#endif
