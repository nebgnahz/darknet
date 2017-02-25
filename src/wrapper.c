#include "wrapper.h"
#include "box.h"
#include "cost_layer.h"
#include "demo.h"
#include "network.h"
#include "option_list.h"
#include "parser.h"
#include "region_layer.h"
#include "utils.h"

extern void cuda_set_device(int n);
extern int gpu_index;

Darknet *darknet_new() {
  cuda_set_device(gpu_index);
  char *datacfg = "cfg/coco.data";
  char *cfgfile = "cfg/yolo.cfg";
  char *weights = "yolo.weights";

  list *options = read_data_cfg(datacfg);
  char *name_list = option_find_str(options, "names", "data/names.list");
  char **names = get_labels(name_list);

  network net = parse_network_cfg(cfgfile);
  load_weights(&net, weights);
  set_batch_network(&net, 1);

  layer l = net.layers[net.n - 1];
  int net_size = l.w * l.h * l.n;
  box *boxes = calloc(net_size, sizeof(box));
  float **probs = calloc(net_size, sizeof(float *));

  for (int j = 0; j < net_size; ++j)
    probs[j] = calloc(l.classes + 1, sizeof(float *));

  Size s = {.width = net.w, .height = net.h};
  Darknet* darknet = calloc(1, sizeof(Darknet));
  darknet->network = net;
  darknet->names = names;
  darknet->boxes = boxes;
  darknet->probs = probs;
  darknet->thresh = 0.24;
  darknet->last_layer = l;
  darknet->nms = 0.4;
  darknet->net_size = net_size;
  darknet->image_size = s;
  return darknet;
}

Rect *darknet_detect(Darknet *darknet, InputImage image) {
  clock_t time;
  time = clock();

  network_predict(darknet->network, image.data);
  printf("Predicted in %f seconds.\n", sec(clock() - time));

  layer l = darknet->last_layer;
  float thresh = darknet->thresh;
  float **probs = darknet->probs;
  char **names = darknet->names;
  box *boxes = darknet->boxes;
  float nms = darknet->nms;
  int net_size = darknet->net_size;

  get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, 0.5);

  if (l.softmax_tree && nms)
    do_nms_obj(boxes, probs, net_size, l.classes, nms);
  else if (nms)
    do_nms_sort(boxes, probs, net_size, l.classes, nms);

  // The network will return l.w * l.h * l.n detections
  // But only those exceeds the threshold are valid
  for (int i = 0; i < net_size; ++i) {
    int class = max_index(probs[i], l.classes);
    float prob = probs[i][class];
    if (prob > thresh) {
      printf("%s: %.0f%%\n", names[class], prob * 100);
      box b = boxes[i];


      printf("%.2f %.2f %.2f %.2f\n", b.x, b.y, b.w, b.h);
    }
  }

  Rect* detections;
  return detections;
}

Size darknet_size(const Darknet *const darknet) {
  return darknet->image_size;
}

void darknet_drop(Darknet *darknet) {
  free(darknet->boxes);
  free_ptrs((void **)darknet->probs, darknet->net_size);
  free(darknet);
}
