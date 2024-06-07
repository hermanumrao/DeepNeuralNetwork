#include <cmath>
// #include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;

double lr = 1; // defining the learning rate
double sigmoid(double x);
vector<int> training_data(int cyc);
double mse_calc(double actual, double pred);
vector<vector<double>> backward_pass(double err, double Z, vector<double> w2,
                                     vector<double> Y);

void printv(vector<double> v) {
  for (int i = 0; i < v.size(); i++)
    cout << v[i] << ' ';
  cout << endl;
}
void printvv(vector<vector<double>> vv) {
  for (int i = 0; i < vv.size(); i++)
    printv(vv[i]);
  cout << endl;
}

class input_l {
private:
  double value;
  vector<double> weights;

public:
  input_l(double val, vector<double> w) {
    value = val;
    weights = w;
  }
  double get_weight(int num) { return weights[num]; }
  void set_weight(int num, double val) { weights[num] = val; }
  void set_value(double val) { value = val; }
  void GrD_update(vector<double> dw) {
    for (int i = 0; i < weights.size(); i++) {
      weights[i] = weights[i] + (lr * dw[i] * value);
    }
  }

  void describe() {
    cout << value << '\t';
    for (int i = 0; i < weights.size(); i++) {
      cout << weights[i] << " ";
    }
    cout << "\t\t input" << endl;
  }
};

class hidden_l {
private:
  double value;
  double act_val;
  vector<double> weights;

public:
  void forward_prop(vector<double> w, vector<double> X) {
    value = 0; // Reset value before accumulation
    try {
      if (w.size() != X.size())
        throw(3);
    } catch (int i) {
      cout << "size of weight vector is not equal to size of vector of inputs"
           << endl;
      exit(0);
    }
    for (int i = 0; i < w.size(); i++) {
      value += w[i] * X[i];
    }
    act_val = sigmoid(value);
  }
  void GrD_update(double dw) {
    for (int i = 0; i < weights.size(); i++) {
      weights[i] = weights[i] + (lr * dw * act_val);
    }
  }
  double backward_pass(double d5) {
    return act_val * (1 - act_val) * weights[0] * d5;
  }
  hidden_l(double val, vector<double> w) {
    value = val;
    act_val = sigmoid(val);
    weights = w;
  }
  void describe() {
    cout << value << '\t' << act_val << '\t';
    for (int i = 0; i < weights.size(); i++) {
      cout << weights[i] << " ";
    }
    cout << "\t hidden" << endl;
  }
  double get_weight(int num) { return weights[num]; }
  double get_act_val() { return act_val; }
  double get_val() { return value; }
};

class output_l {
private:
  double value;
  double act_val;

public:
  void forward_prop(vector<double> w, vector<double> Y) {
    value = 0; // Reset value before accumulation
    try {
      if (w.size() != Y.size())
        throw(3);
    } catch (int i) {
      cout << "size of weight vector is not equal to size of vector of inputs"
           << endl;
      exit(0);
    }
    for (int i = 0; i < w.size(); i++) {
      value += w[i] * Y[i];
    }
    act_val = sigmoid(value);
  }

  double final_value() { return act_val; }

  double backward_pass(double actl) {
    return (actl - act_val) * act_val * (1 - act_val);
  }

  output_l(double val) {
    value = val;
    act_val = sigmoid(val);
  }
  double get_val() { return value; }
  void describe() { cout << value << '\t' << act_val << "\t\t out" << endl; }
};

vector<input_l> init_input(int cnt);
vector<hidden_l> init_hidden(int cnt);
vector<output_l> init_output(int cnt);

int main(int argc, char *argv[]) {
  // INSTANTIATING ALL 3 TYPES OF LAYERS
  vector<input_l> in_layer = init_input(2);
  vector<hidden_l> hid_layer = init_hidden(2);
  vector<output_l> out_layer = init_output(1);

  int cycles;
  cout << "how many cycles of training to run?" << endl;
  cin >> cycles;

  for (int cyc = 0; cyc < cycles; cyc++) {
    cout << "cycle:" << cyc + 1 << endl;
    vector<int> data = training_data(cyc);
    for (int i = 0; i < in_layer.size(); i++) { // setting up the input layer
      in_layer[i].set_value(
          (double)data[i]); // feeding the layer with training data
    }
    for (int i = 0; i < hid_layer.size(); i++) { // setting up the hidden layer
      vector<double> w1, X;
      for (int j = 0; j < in_layer.size();
           j++) { // getting weights and x.val from input layer
        w1.push_back(in_layer[j].get_weight(i));
        X.push_back((double)data[j]);
      }
      hid_layer[i].forward_prop(w1, X);
    }
    for (int i = 0; i < out_layer.size(); i++) { // setting up the hidden layer
      vector<double> w2, Y;
      for (int j = 0; j < hid_layer.size();
           j++) { // getting weights and x.val from input layer
        w2.push_back(hid_layer[j].get_weight(i));
        Y.push_back(hid_layer[j].get_act_val());
      }
      out_layer[i].forward_prop(w2, Y);
    }
    for (int i = 0; i < in_layer.size(); i++) {
      in_layer[i].describe();
    }
    for (int i = 0; i < hid_layer.size(); i++) {
      hid_layer[i].describe();
    }
    for (int i = 0; i < out_layer.size(); i++) {
      out_layer[i].describe();
    }
    ////////////////////////////////////////////////////////////
    double Z = data[2];
    double Zd = out_layer[0].final_value();
    double L = mse_calc(Zd, Z); // this helps calculate the loss commented
                                // since not reqired right now
    vector<double> w2, H, X, D;

    for (int i = 0; i < hid_layer.size(); i++) {
      H.push_back(hid_layer[i].get_act_val());
      w2.push_back(hid_layer[i].get_weight(0));
    }
    X = {(double)data[0], (double)data[1]};

    for (int i = 0; i < out_layer.size(); i++) {
      D.push_back(out_layer[i].backward_pass(Z));
    }
    for (int i = 0; i < hid_layer.size(); i++) {
      D.push_back(hid_layer[i].backward_pass(D[0]));
    }
    // now D has values 𝛿5, 𝛿3, 𝛿4 in indexes 0,1,2
    for (int i = 0; i < hid_layer.size(); i++) {
      hid_layer[i].GrD_update(D[0]);
    }
    for (int i = 0; i < in_layer.size(); i++) {
      in_layer[i].GrD_update({D[1], D[2]});
    }

    for (int i = 0; i < in_layer.size(); i++) {
      in_layer[i].describe();
    }
    for (int i = 0; i < hid_layer.size(); i++) {
      hid_layer[i].describe();
    }
    for (int i = 0; i < out_layer.size(); i++) {
      out_layer[i].describe();
    }
  }

  return 0;
}

// Function to compute the sigmoid of a given input
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

// Functions to initialize all the layers
vector<input_l> init_input(int cnt) {
  vector<input_l> lay;
  for (int i = 0; i < cnt; i++) {
    vector<double> weight = {0.2, 0.3};
    input_l inp(0.0, weight);
    lay.push_back(inp);
  }
  return lay;
}

vector<hidden_l> init_hidden(int cnt) {
  vector<hidden_l> lay;
  for (int i = 0; i < cnt; i++) {
    vector<double> weight = {0.3 + i * 0.6};
    hidden_l hid(0.0, weight);
    lay.push_back(hid);
  }
  return lay;
}

vector<output_l> init_output(int cnt) {
  vector<output_l> lay;
  for (int i = 0; i < cnt; i++) {
    output_l out(0.0);
    lay.push_back(out);
  }
  return lay;
}

vector<int> training_data(int cyc) {
  vector<vector<int>> data = {{0, 0, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 0}};
  return data[cyc % 4];
  // return {0.35, 0.7, 0.5};
}

double mse_calc(double actl, double pred) { // for calculating loss
  return 0.5 * (pred - actl) * (pred - actl);
}
