#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

double activation_fn(double x) { return 1.0 / (1.0 + exp(-x)); }

class neuron {
protected:
  double value;
  vector<double> weights;

public:
  neuron(double val, vector<double> ws) {
    value = val;
    weights = ws;
  }
  double get_val() { return value; }
  vector<double> get_weights() { return weights; }
  double get_weight(int i) { return weights[i]; }
  double activate() { return activation_fn(value); }
  void set_value(double nval) { value = nval; }
  void set_weight(int i, double nw) { weights[i] = nw; }
};

class input_layer {
private:
  vector<neuron> layer;

public:
  input_layer(int cnt, vector<double> vals, vector<vector<double>> ws) {
    for (int i = 0; i < cnt; i++) {
      neuron n(vals[i], ws[i]);
      layer.push_back(n);
    }
  }
};

class hidden_layer {
private:
  vector<neuron> layer;
  vector<double> activated_value;

public:
  hidden_layer(int cnt, vector<double> vals, vector<vector<double>> ws) {
    for (int i = 0; i < cnt; i++) {
      neuron n(vals[i], ws[i]);
      layer.push_back(n);
      activated_value.push_back(n.activate());
    }
  }
  void forward_propagate(vector<neuron> inp) {
    for (int i; i < layer.size(); i++) {
      for (int j; j < inp[0].get_weights().size(); j++) {
        layer[i].set_value(inp[i]); // continue here
      }
    }
  }
};

int main(int argc, char *argv[]) {
  int epochs;
  cout << "input number of epochs";
  cin >> epochs;
  for (int epoch = 0; epoch < epochs; epoch++) {
    ;
  }

  return 0;
}
