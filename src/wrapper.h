// This wrapper provides `Darknet` struct as the simple and sole API to simplify
// language bindings (Rust for me).
//
// A side goal here is to create wrapper structs that are independent of darknet
// source (such as Size, Rect, etc).

#include <stdint.h>

#include "network.h"
#include "layer.h"
#include "image.h"

typedef struct {
    int32_t width;
    int32_t height;
} Size;

typedef struct {
    float x;
    float y;
    float w;
    float h;
} Rect;

typedef struct {
    Rect* rects;
    char** labels;
    float* probs;
    int32_t num;
    float proc_time_in_ms;
} Detections;

typedef struct {
  network network;
  int32_t net_size;
  char **names;
  box *boxes;
  float **probs;
  float thresh;
  layer last_layer;
  float nms;
  Size image_size;
} Darknet;

// The input must be resized to the proper size
typedef struct {
  const float *const data;
  int32_t size;
} InputImage;

Darknet* darknet_new();
void darknet_drop(Darknet* darknet);
Size darknet_size(const Darknet *const darknet);
Detections darknet_detect(Darknet *darknet, image image);
void detections_drop(Detections detections);
