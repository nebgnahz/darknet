#include <assert.h>

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

Darknet *darknet_new(DarknetConfig dn_config) {
  cuda_set_device(gpu_index);
  char *datacfg = dn_config.datacfg;
  char *cfgfile = dn_config.network_file;
  char *weights = dn_config.weight_file;

  list *options = read_data_cfg(datacfg);
  char *name_list = option_find_str(options, "names", dn_config.label_file);
  char **names = get_labels(name_list);

  network net = parse_network_cfg(cfgfile);
  load_weights(&net, weights);
  set_batch_network(&net, 1);

  layer l = net.layers[net.n - 1];
  int32_t net_size = l.w * l.h * l.n;
  box *boxes = calloc(net_size, sizeof(box));
  float **probs = calloc(net_size, sizeof(float *));

  for (int32_t j = 0; j < net_size; ++j)
    probs[j] = calloc(l.classes + 1, sizeof(float *));

  Darknet* darknet = calloc(1, sizeof(Darknet));
  darknet->network = net;
  darknet->names = names;
  darknet->boxes = boxes;
  darknet->probs = probs;
  darknet->thresh = 0.24;
  darknet->last_layer = l;
  darknet->nms = 0.4;
  darknet->net_size = net_size;
  return darknet;
}

Detections darknet_detect(Darknet *darknet, image im) {
  clock_t time;
  Detections detections;
  image sized = resize_image(im, darknet->network.w, darknet->network.h);
  float *X = sized.data;

  time = clock();
  network_predict(darknet->network, X);
  detections.proc_time_in_ms = sec(clock() - time);

  layer l = darknet->last_layer;
  float thresh = darknet->thresh;
  float **probs = darknet->probs;
  char **names = darknet->names;
  box *boxes = darknet->boxes;
  float nms = darknet->nms;
  int32_t net_size = darknet->net_size;

  get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, 0.5);

  if (l.softmax_tree && nms)
    do_nms_obj(boxes, probs, net_size, l.classes, nms);
  else if (nms)
    do_nms_sort(boxes, probs, net_size, l.classes, nms);

  // The network will return l.w * l.h * l.n detections
  // But only those exceeds the threshold are valid
  int32_t num = 0;

  // First count how many
  for (int32_t i = 0; i < net_size; ++i) {
    int32_t class = max_index(probs[i], l.classes);
    float prob = probs[i][class];
    if (prob > thresh) {
      num++;
    }
  }

  detections.rects = calloc(num, sizeof(Rect));
  detections.labels = calloc(num, sizeof(char*));
  detections.probs = calloc(num, sizeof(float));
  detections.num = num;

  // reset num and start to assign
  num = 0;
  for (int32_t i = 0; i < net_size; ++i) {
    int32_t class = max_index(probs[i], l.classes);
    float prob = probs[i][class];
    if (prob > thresh) {
      detections.labels[num] = names[class];
      detections.probs[num] = prob;
      box b = boxes[i];
      detections.rects[num].x = b.x;
      detections.rects[num].y = b.y;
      detections.rects[num].w = b.w;
      detections.rects[num].h = b.h;

      num++;
    }
  }
  assert(num == detections.num);
  return detections;
}

void darknet_drop(Darknet *darknet) {
  free(darknet->boxes);
  free_ptrs((void **)darknet->probs, darknet->net_size);
  free(darknet);
}

void detections_drop(Detections detections) {
  free(detections.rects);
  free(detections.probs);
  free(detections.labels);
}
