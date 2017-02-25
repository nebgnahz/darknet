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

// Run this one video
// ./darknet api <image file>
//
void detect_api(char *filename) {
  cuda_set_device(gpu_index);

  char *datacfg = "cfg/coco.data";
  char *cfgfile = "cfg/yolo.cfg";
  char *weights = "yolo.weights";

  list *options = read_data_cfg(datacfg);
  char *name_list = option_find_str(options, "names", "data/names.list");
  char **names = get_labels(name_list);

  image **alphabet = load_alphabet();
  network net = parse_network_cfg(cfgfile);
  load_weights(&net, weights);
  set_batch_network(&net, 1);

  srand(2222222);
  clock_t time;
  char buff[256];
  char *input = buff;
  int j;
  float nms = .4;

  image im = load_image_color(filename, 0, 0);
  image sized = resize_image(im, net.w, net.h);
  layer l = net.layers[net.n - 1];

  box *boxes = calloc(l.w * l.h * l.n, sizeof(box));
  float **probs = calloc(l.w * l.h * l.n, sizeof(float *));
  for (j = 0; j < l.w * l.h * l.n; ++j)
    probs[j] = calloc(l.classes + 1, sizeof(float *));

  float *X = sized.data;
  float thresh = 0.24;
  float hier_thresh = .5;

  time = clock();
  network_predict(net, X);
  printf("%s: Predicted in %f seconds.\n", input, sec(clock() - time));
  get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, 0.5);
  if (l.softmax_tree && nms)
    do_nms_obj(boxes, probs, l.w * l.h * l.n, l.classes, nms);
  else if (nms)
    do_nms_sort(boxes, probs, l.w * l.h * l.n, l.classes, nms);
  draw_detections(im, l.w * l.h * l.n, thresh, boxes, probs, names, alphabet,
                  l.classes);
  save_image(im, "predictions");
  show_image(im, "predictions");

  free_image(im);
  free_image(sized);
  free(boxes);
  free_ptrs((void **)probs, l.w * l.h * l.n);
}
