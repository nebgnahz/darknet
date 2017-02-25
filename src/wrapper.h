// This wrapper provides `Darknet` struct as the simple and sole API to simplify
// language bindings (Rust for me).
//
// A side goal here is to create wrapper structs that are independent of darknet
// source (such as Size, Rect, etc).

#include "network.h"
#include "layer.h"

typedef struct {
    int width;
    int height;
} Size;

typedef struct {
    float x;
    float y;
    float w;
    float h;
} Rect;

typedef struct {
  network network;
  int net_size;
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
  float *data;
  int size;
} InputImage;

Darknet* darknet_new();
void darknet_drop(Darknet* darknet);
Size darknet_size(const Darknet *const darknet);
Rect *darknet_detect(Darknet *darknet, InputImage image);
